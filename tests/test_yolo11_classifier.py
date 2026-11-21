import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
from PIL import Image
import pytest
import yaml

from training.classifiers import yolo11
from training.splitting import grouped_folds


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    classes = ["species_b", "species_a"]
    rows, review = [], []
    for label in classes:
        directory = root / "images" / label
        directory.mkdir(parents=True)
        for index in range(10):
            parent = f"{label}/{index}.tif"
            crops = []
            for instance in range(2):
                crop_id = f"{index}_{instance}"
                crops.append(dict(crop_id=crop_id, active=True, decision="Accept target"))
                rows.append(dict(image_id=f"{label}/{crop_id}.png", parent_image_id=parent))
                Image.new("RGB", (40, 8), "red").save(directory / f"{crop_id}.png")
            review.append(dict(reviewed=True, target=label, parent_image_id=parent, crops=crops))
    pd.DataFrame(rows).to_csv(root / "parent_groups.csv", index=False)
    (root / "training.yaml").write_text(yaml.safe_dump({"dataset": {"classes": classes}}))
    yolo11.write_json(root / "review_snapshot.json", {"collection": "training", "images": review})
    return root


@pytest.mark.parametrize("change", ["evaluation", "unreviewed", "parent", "corrupt"])
def test_snapshot_rejects_unverified_crops(snapshot, change):
    path = snapshot / "review_snapshot.json"
    review = json.loads(path.read_text())
    if change == "evaluation":
        review["collection"] = "evaluation"
    elif change == "unreviewed":
        review["images"][0]["reviewed"] = False
    elif change == "parent":
        review["images"][0]["parent_image_id"] = "different"
    else:
        (snapshot / "images/species_b/0_0.png").write_bytes(b"broken")
    yolo11.write_json(path, review)
    with pytest.raises(ValueError):
        yolo11.load_snapshot(snapshot)


def test_folds_and_crop_padding(snapshot, tmp_path):
    manifest, _ = yolo11.load_snapshot(snapshot)
    baseline = grouped_folds(manifest.Class, manifest.parent_image_id, 5, 42, "Baseline")
    for (train, val, test), (baseline_train, baseline_test) in zip(yolo11.plan_folds(manifest), baseline):
        assert sorted(test) == sorted(baseline_test)
        assert sorted([*train, *val]) == sorted(baseline_train)
        train_groups, val_groups, test_groups = (set(manifest.iloc[i].parent_image_id) for i in (train, val, test))
        assert not (train_groups & val_groups or val_groups & test_groups or train_groups & test_groups)

    staged = yolo11.stage_crops(snapshot / "images", tmp_path / "staged", ["species_b/0_0.png"])
    with Image.open(next(iter(staged))) as image:
        assert image.mode == "RGB" and image.size == (224, 224)
        assert image.getbbox() == (0, 89, 224, 134)


@pytest.mark.parametrize("fail_fold", [None, 2])
def test_training_reports_and_retains_completed_folds(snapshot, tmp_path, monkeypatch, fail_fold):
    weights = tmp_path / yolo11.MODEL
    weights.write_bytes(b"starting weights")
    output = tmp_path / "output"

    class Model:
        task = "classify"

        def __init__(self, path):
            self.ckpt_path = path

        def train(self, *, data, project, name, **settings):
            assert self.ckpt_path == str(weights)
            assert settings["epochs"] == 1 and settings["scale"] == 0
            assert not (Path(data) / "test").exists()
            if Path(project).name == f"outer_{fail_fold}":
                raise RuntimeError("fold interrupted")
            best = Path(project) / name / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.touch()
            self.trainer = SimpleNamespace(best=best)

        def predict(self, *, source, **settings):
            assert self.ckpt_path.endswith("best.pt")
            for path in reversed(source):
                yield SimpleNamespace(
                    path=path, names={0: "species_a", 1: "species_b"},
                    probs=SimpleNamespace(top1=int(Path(path).parent.name == "species_b")),
                )

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    monkeypatch.setattr(yolo11, "version", lambda package: "test")
    if fail_fold:
        with pytest.raises(RuntimeError, match="fold interrupted"):
            yolo11.train(snapshot, output, epochs=1, device="cpu", weights=weights)
        status = json.loads((output / "run_status.json").read_text())
        assert status["status"] == "failed" and status["completed_folds"] == 1
        assert (output / "outer_1/test_predictions.csv").is_file()
        assert not (output / "summary.json").exists()
        return

    yolo11.train(snapshot, output, epochs=1, device="cpu", weights=weights)
    assert json.loads((output / "run_status.json").read_text())["status"] == "completed"
    predictions = pd.read_csv(output / "oof_predictions.csv")
    confusion = pd.read_csv(output / "confusion_matrix.csv", index_col=0)
    assert len(predictions) == predictions.image_id.nunique() == confusion.values.sum() == 40
    assert (predictions.Class == predictions.prediction).all()
    assert list(confusion.index) == list(confusion.columns) == ["species_b", "species_a"]
    assert json.loads((output / "summary.json").read_text())["pooled_oof_metrics"]["macro_f1"] == 1
    assert (output / "summary.md").is_file()
    assert (output / "confusion_matrix.png").is_file()
    assert len(list(output.glob("outer_*/fit/weights/best.pt"))) == 5
