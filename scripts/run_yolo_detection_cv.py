"""Train and evaluate a full-field YOLO detector with grouped cross-validation."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import yaml
import numpy as np

from scripts._detection_references import source_path
from scripts.run_detection_pilot import source_groups
from training.splitting import grouped_folds


SEED = 42
FOLD_COUNT = 5
METRICS = ("precision", "recall", "AP50", "AP50-95")


def _write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=np.generic.item) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def plan_folds(rows):
    labels = [row["target"] for row in rows]
    sources = {row["image_id"]: {"parent_image_id": row["parent_image_id"], "specimen_id": ""} for row in rows}
    groups = [0] * len(rows)
    for group in source_groups(rows, sources):
        group_id = min(group)
        for index in group:
            groups[index] = group_id
    folds = []
    for outer_train, test in grouped_folds(labels, groups, FOLD_COUNT, SEED, "Detection outer CV"):
        train, val = grouped_folds([labels[i] for i in outer_train], [groups[i] for i in outer_train],
                                   FOLD_COUNT, SEED, "Detection inner validation")[0]
        folds.append(dict(train=outer_train[train], val=outer_train[val], test=test))
    if Counter(i for fold in folds for i in fold["test"]) != dict.fromkeys(range(len(rows)), 1):
        raise ValueError("Each field must appear in exactly one outer test fold")
    return folds


def _load_dataset(dataset):
    names = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))["names"]
    with (dataset / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [dict(row, sha256=row["source_sha256"]) for row in csv.DictReader(stream) if row["split"]]
    rows.sort(key=lambda row: row["image_id"])
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Dataset manifest has duplicate field IDs")
    if set(row["target"] for row in rows) != set(names.values()):
        raise ValueError("Dataset manifest and YOLO class names differ")
    images = []
    for row in rows:
        if not row["parent_image_id"] or not row["sha256"]:
            raise ValueError(f"Missing group identity for {row['image_id']}")
        image = source_path(dataset, row["image_file"])
        label = dataset / "labels" / row["split"] / (image.stem + ".txt")
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing exported image or label for {row['image_id']}")
        images.append(image)
    return names, rows, images


def _write_fold_file(path, content):
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing fold configuration changed: {path}")
        return
    path.write_text(content, encoding="utf-8")


def _prepare_fold(fold_dir, dataset, names, images, fold):
    fold_dir.mkdir(exist_ok=True)
    config = dict(path=dataset.as_posix(), names=names)
    for split, indices in fold.items():
        paths = fold_dir / f"{split}.txt"
        _write_fold_file(paths, "".join(f"{images[i].as_posix()}\n" for i in indices))
        config[split] = paths.as_posix()
    path = fold_dir / "data.yaml"
    _write_fold_file(path, yaml.safe_dump(config, sort_keys=False))
    return path


def _score_fold(model_type, weights, data, fold_dir, settings):
    fit = fold_dir / "fit"
    checkpoint = fit / "weights" / "best.pt"
    last = fit / "weights" / "last.pt"
    progress = fit / "results.csv"
    completed = False
    if checkpoint.is_file() and progress.is_file():
        with progress.open(encoding="utf-8", newline="") as stream:
            epochs = [int(row["epoch"]) for row in csv.DictReader(stream)]
        completed = bool(epochs) and epochs[-1] == settings["epochs"]
    if completed:
        print(f"Scoring the existing completed checkpoint: {checkpoint}", flush=True)
    else:
        model = model_type(str(last if last.is_file() else weights))
        if model.task != "detect":
            raise ValueError("Initial weights must be a YOLO detection checkpoint")
        if last.is_file():
            model.train(resume=True, device=settings["device"])
        else:
            model.train(data=str(data), project=str(fold_dir), name="fit", exist_ok=True, **settings)
        checkpoint = Path(model.trainer.best)
    if not checkpoint.is_file():
        raise RuntimeError(f"Training did not save a best checkpoint in {fold_dir}")
    model = model_type(str(checkpoint))
    evaluation = {key: settings[key] for key in ("batch", "device", "imgsz", "workers")}
    metrics = model.val(data=str(data), split="test", project=str(fold_dir), name="test",
                        plots=False, **evaluation)
    return checkpoint, dict(zip(METRICS, map(float, metrics.mean_results()))), metrics.summary(decimals=6)


def run(dataset, output, weights, *, epochs=50, device="0", resume=False):
    from ultralytics import YOLO, __version__
    import torch

    dataset, output, weights = (Path(path).resolve() for path in (dataset, output, weights))
    if not weights.is_file():
        raise FileNotFoundError(f"Local pretrained weights not found: {weights}")
    names, rows, images = _load_dataset(dataset)
    folds = plan_folds(rows)
    settings = dict(epochs=epochs, imgsz=640, batch=2, device=device, workers=0, seed=0)
    config = dict(dataset=str(dataset), weights=str(weights), ultralytics=__version__, torch=torch.__version__,
                  manifest_sha256=sha256((dataset / "manifest.csv").read_bytes()).hexdigest(),
                  weights_sha256=sha256(weights.read_bytes()).hexdigest(), split_seed=SEED, training=settings)
    if output.is_relative_to(dataset):
        raise ValueError("Choose an output directory outside the input dataset")
    if resume:
        previous = json.loads((output / "settings.json").read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Run settings or inputs changed. Choose a new output directory.")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "settings.json", config)
    configs = []
    for number, fold in enumerate(folds, start=1):
        fold_dir = output / f"fold_{number:02d}"
        configs.append(_prepare_fold(fold_dir, dataset, names, images, fold))
    with (output / "fold_assignments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fold", "split", "image_id", "parent_image_id", "target"])
        for number, fold in enumerate(folds, start=1):
            for split, indices in fold.items():
                for i in indices:
                    row = rows[i]
                    writer.writerow([number, split, row["image_id"], row["parent_image_id"], row["target"]])

    results = []
    for number, (fold, data) in enumerate(zip(folds, configs), start=1):
        metrics_path = data.parent / "test_metrics.json"
        if metrics_path.is_file():
            results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            continue
        counts = {name: len(indices) for name, indices in fold.items()}
        print(f"Starting fold {number}/{FOLD_COUNT}: {counts}", flush=True)
        checkpoint, aggregate, species = _score_fold(YOLO, weights, data, data.parent, settings)
        if {row["Class"] for row in species} != set(names.values()):
            raise RuntimeError(f"Fold {number} test metrics are missing a target species")
        report = dict(fold=number, fields=counts, checkpoint=str(checkpoint), metrics=aggregate, species=species)
        _write_json(metrics_path, report)
        results.append(report)
        print(f"Completed fold {number}/{FOLD_COUNT}: {aggregate}", flush=True)

    summary = {}
    for key in METRICS:
        values = [result["metrics"][key] for result in results]
        summary[key] = dict(mean=mean(values), sample_stdev=stdev(values))
    _write_json(output / "summary.json", dict(folds=results, summary=summary))
    print(f"Cross-validation complete. Report: {output / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.dataset_dir, args.output_dir, args.weights, epochs=args.epochs, device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
