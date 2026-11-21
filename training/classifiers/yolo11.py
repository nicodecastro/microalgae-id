"""Train and evaluate YOLO11s on an exported reviewed-crop snapshot."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report
import yaml

from microalgae.imaging.dataset import build_dataset_manifest
from training.evaluation import _save_confusion, _scores, write_json
from training.splitting import grouped_folds, load_parent_groups


MODEL = "yolo11s-cls.pt"
IMAGE_SIZE = 224
SEED = 42
METRICS = {
    "macro_f1": "Macro-F1", "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy", "macro_precision": "Macro precision",
    "macro_recall": "Macro recall",
}


def load_snapshot(snapshot):
    config = yaml.safe_load((snapshot / "training.yaml").read_text(encoding="utf-8"))
    classes = config["dataset"]["classes"]
    if not isinstance(classes, list) or any(
        not isinstance(label, str) or Path(label).name != label or label in {".", ".."}
        for label in classes
    ):
        raise ValueError("Snapshot classes must be a list of class folder names")
    images = snapshot / "images"
    manifest = load_parent_groups(
        build_dataset_manifest(images, classes), snapshot / "parent_groups.csv",
    )
    review = json.loads((snapshot / "review_snapshot.json").read_text(encoding="utf-8"))
    if review.get("collection") != "training":
        raise ValueError("A reviewed training collection is required")

    accepted = {}
    for source in review["images"]:
        if source["reviewed"] is not True:
            continue
        for crop in source["crops"]:
            if crop["active"] is True and crop["decision"] == "Accept target":
                image_id = f"{source['target']}/{crop['crop_id']}.png"
                if image_id in accepted:
                    raise ValueError(f"Duplicate accepted annotation: {image_id}")
                accepted[image_id] = source["parent_image_id"]
    if accepted != dict(zip(manifest.image_id, manifest.parent_image_id)):
        raise ValueError("Reviewed annotations do not match the exported crops and parent mapping")
    if set(manifest.Class) != set(classes):
        raise ValueError("Every configured class needs reviewed crops")

    for image_id in manifest.image_id:
        try:
            with Image.open(images / image_id) as image:
                image.load()
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read reviewed crop: {image_id}") from error
    return manifest, classes


def plan_folds(manifest):
    folds = []
    for number, (outer_train, test) in enumerate(
        grouped_folds(manifest.Class, manifest.parent_image_id, 5, SEED, "YOLO outer CV"), 1
    ):
        training = manifest.iloc[outer_train]
        train, val = grouped_folds(
            training.Class, training.parent_image_id, 5, SEED, f"YOLO fold {number} validation",
        )[0]
        folds.append((outer_train[train], outer_train[val], test))
    return folds


def stage_crops(images, destination, image_ids):
    paths = {}
    for image_id in image_ids:
        path = destination / image_id
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(images / image_id) as image:
            scale = IMAGE_SIZE / max(image.size)
            size = tuple(max(1, round(length * scale)) for length in image.size)
            resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
            square = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
            square.paste(resized, tuple((IMAGE_SIZE - length) // 2 for length in size))
            square.save(path)
        paths[path.resolve()] = image_id
    return paths


def _run_fold(model_factory, weights, images, directory, train, val, test, classes, settings):
    directory.mkdir()
    with TemporaryDirectory(dir=directory) as temporary:
        dataset = Path(temporary)
        stage_crops(images, dataset / "train", train.image_id)
        stage_crops(images, dataset / "val", val.image_id)
        model = model_factory(weights)
        if model.task != "classify":
            raise ValueError("The starting weights must be YOLO11s classification weights")
        model.train(data=str(dataset), project=str(directory), name="fit", **settings)
        checkpoint = Path(model.trainer.best)
        if not checkpoint.is_file():
            raise RuntimeError("Training did not produce a best validation checkpoint")

        paths = stage_crops(images, dataset / "test", test.image_id)
        predictions = {}
        fitted = model_factory(str(checkpoint))
        sources = [str(path) for path in paths]
        batch = settings["batch"]
        for start in range(0, len(sources), batch):
            for result in fitted.predict(
                source=sources[start:start + batch], imgsz=IMAGE_SIZE, batch=batch,
                device=settings["device"], verbose=False, stream=True,
            ):
                image_id = paths.pop(Path(result.path).resolve(), None)
                if image_id is None or result.probs is None or set(result.names.values()) != set(classes):
                    raise ValueError("Unexpected test prediction, image path, or model classes")
                predictions[image_id] = result.names[result.probs.top1]
        if paths:
            raise ValueError("The checkpoint did not predict every test crop")
    return test.image_id.map(predictions).to_numpy()


def _report(output, predictions, fold_metrics, classes):
    expected, predicted = predictions.Class, predictions.prediction
    per_species = classification_report(
        expected, predicted, labels=classes, output_dict=True, zero_division=0,
    )
    per_species = {label: per_species[label] for label in classes}
    metrics = pd.DataFrame(fold_metrics).drop(columns="fold")
    summary = {
        "task": "Species classification on reviewed target crops",
        "outer_fold_summary": {
            key: {"mean": metrics[key].mean(), "std_ddof0": metrics[key].std(ddof=0)}
            for key in METRICS
        },
        "pooled_oof_metrics": _scores(expected, predicted, classes),
        "per_species": per_species,
    }
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    _save_confusion(expected, predicted, classes, output)
    write_json(output / "summary.json", summary)

    lines = [
        "# YOLO11s crop classification", "",
        f"Species classification on {len(predictions):,} reviewed target crops "
        f"from {predictions.parent_image_id.nunique():,} parents.", "",
        "| Metric | Pooled out-of-fold | Fold mean | Fold SD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in METRICS.items():
        stats = summary["outer_fold_summary"][key]
        pooled = summary["pooled_oof_metrics"][key]
        lines.append(f"| {label} | {pooled:.2%} | {stats['mean']:.2%} | {stats['std_ddof0']:.2%} |")
    lines.extend(["", "| Species | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |"])
    for label, scores in per_species.items():
        lines.append(
            f"| {label} | {scores['precision']:.2%} | {scores['recall']:.2%} | "
            f"{scores['f1-score']:.2%} | {int(scores['support'])} |"
        )
    lines.extend(["", "![Pooled out-of-fold confusion matrix](confusion_matrix.png)", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def train(snapshot, output, *, epochs=50, device="auto", weights=None):
    if output.is_relative_to(snapshot) or snapshot.is_relative_to(output):
        raise ValueError("The output directory must be separate from the reviewed snapshot")
    if output.exists():
        raise FileExistsError(f"Output directory must be new: {output}")
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    manifest, classes = load_snapshot(snapshot)
    folds = plan_folds(manifest)
    if weights is not None and not weights.is_file():
        raise FileNotFoundError(f"Local classification weights do not exist: {weights}")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError('Install the optional YOLO dependency with python -m pip install -e ".[yolo]"') from error
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    settings = dict(
        epochs=epochs, imgsz=IMAGE_SIZE, batch=8, patience=10, workers=0,
        seed=SEED, deterministic=True, device=device, amp=False, scale=0.0,
        auto_augment=None, erasing=0.0, fliplr=0.0, flipud=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, plots=False,
    )

    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "completed_folds": 0}

    def update_status():
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run_status.json", status)

    update_status()
    try:
        # Resolve the official weights once into the run directory, then start each fold afresh.
        initial = YOLO(str(weights) if weights is not None else str(output / MODEL))
        starting_weights = Path(initial.ckpt_path).resolve()
        weights_hash = sha256(starting_weights.read_bytes()).hexdigest()
        write_json(output / "settings.json", {
            "model": MODEL, "weights": starting_weights, "weights_sha256": weights_hash,
            "snapshot": snapshot, "classes": classes, "padding": "centered black RGB",
            "training": settings, "ultralytics": version("ultralytics"), "torch": version("torch"),
            "scikit_learn": version("scikit-learn"),
        })
        split_rows = []
        for fold, partitions in enumerate(folds, 1):
            for role, indices in zip(("train", "val", "test"), partitions):
                split_rows.append(manifest.iloc[indices].assign(outer_fold=fold, role=role))
        pd.concat(split_rows, ignore_index=True).to_csv(output / "splits.csv", index=False)

        predictions = []
        fold_metrics = []
        for fold, (train_ids, val_ids, test_ids) in enumerate(folds, 1):
            status["current_fold"] = fold
            update_status()
            print(f"YOLO11s outer fold {fold}/5", flush=True)
            directory = output / f"outer_{fold}"
            train_rows, val, test = (manifest.iloc[indices] for indices in (train_ids, val_ids, test_ids))
            predicted = _run_fold(
                YOLO, str(starting_weights), snapshot / "images", directory,
                train_rows, val, test, classes, settings,
            )
            result = test.assign(prediction=predicted, outer_fold=fold)
            result.to_csv(directory / "test_predictions.csv", index=False)
            predictions.append(result)
            fold_metrics.append({"fold": fold, **_scores(test.Class, predicted, classes)})
            pd.DataFrame(fold_metrics).to_csv(output / "fold_metrics.csv", index=False)
            status["completed_folds"] = fold
            update_status()

        pooled = pd.concat(predictions).sort_index()
        if not np.array_equal(pooled.image_id.to_numpy(), manifest.image_id.to_numpy()):
            raise ValueError("Expected exactly one outer-test prediction per reviewed crop")
        _report(output, pooled, fold_metrics, classes)
        status.update(status="completed", current_fold=None)
        update_status()
    except (Exception, KeyboardInterrupt) as error:
        status.update(status="failed", reason=str(error), traceback=traceback.format_exc())
        update_status()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device index")
    parser.add_argument("--weights", type=Path, help="Local copy of the official yolo11s-cls.pt")
    args = parser.parse_args()
    try:
        train(
            args.snapshot_dir.resolve(), args.output_dir.resolve(), epochs=args.epochs,
            device=args.device, weights=args.weights.resolve() if args.weights else None,
        )
    except Exception as error:
        parser.exit(1, f"YOLO11s training failed: {error}\n")
    print(f"Report: {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
