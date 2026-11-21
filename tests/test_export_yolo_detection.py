import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from microalgae.imaging.crop_review import TARGET_CLASSES, read_image
from scripts.export_yolo_detection import export_dataset


@pytest.fixture
def snapshot(tmp_path):
    raw = tmp_path / "raw"
    batch = tmp_path / "batch"
    sources = batch / "sources"
    directory = tmp_path / "snapshot"
    for path in (raw, sources, directory):
        path.mkdir(parents=True)
    rows = []
    for class_id, species in enumerate(TARGET_CLASSES):
        for field in range(3):
            name = f"field-{class_id}-{field}.jpg"
            local = class_id == field == 0
            source = (sources if local else raw) / name
            image = Image.new("RGB", (40, 20), (30 + class_id * 70, 20 + field * 70, 20))
            exif = image.getexif()
            exif[274] = 6
            image.save(source, exif=exif)
            rows.append(dict(
                image_id=name, parent_image_id=name, target=species, status="proposals",
                reviewed=True, sha256=sha256(source.read_bytes()).hexdigest(),
                batch_local_source=local, source_path=f"sources/{name}" if local else name,
                pilot_partition="evaluation" if local else None,
                crops=[dict(active=True, decision="Accept target", box=[4, 2, 20, 12])],
            ))

    data = dict(schema_version=1, collection="training", root=str(raw), images=rows)
    return directory, batch, data


def _save(directory, data):
    (directory / "review_snapshot.json").write_text(json.dumps(data), encoding="utf-8")


def _manifest(output):
    with (output / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        return {row["image_id"]: row for row in csv.DictReader(stream)}


def test_export_labels_original_orientation_and_review_decisions(snapshot, tmp_path):
    directory, batch, data = snapshot
    local = data["images"][0]
    local["crops"] += [dict(active=True, decision="Non-target alga", box=[22, 2, 28, 10]),
                       dict(active=False, decision="Accept target", box=[0, 0, 40, 20]),
                       dict(active=False, decision="Unsure / mixed", box=[0, 0, 40, 20])]
    data["images"].extend([
        dict(local, image_id="ambiguous.jpg", crops=local["crops"] + [dict(active=True, decision="Unsure / mixed")]),
        dict(local, image_id="pending.jpg", reviewed=False),
        dict(local, image_id="non-target.jpg", crops=[dict(active=True, decision="Non-target alga")]),
    ])
    _save(directory, data)
    output = tmp_path / "dataset"
    counts = export_dataset(directory, output, batch_dir=batch)
    assert sum(counts.values()) == 9
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config == dict(path=output.as_posix(), train="images/train", val="images/val",
                          names=dict(enumerate(TARGET_CLASSES)))
    manifest = _manifest(output)
    row = manifest[local["image_id"]]
    assert row["pilot_partition"] == "evaluation"
    assert row["box_count"] == "1"
    image_file = output / row["image_file"]
    label_file = output / "labels" / row["split"] / (image_file.stem + ".txt")
    assert label_file.read_text(encoding="utf-8") == "0 0.3 0.35 0.4 0.5\n"
    assert read_image(image_file).shape == (20, 40, 3)
    np.testing.assert_array_equal(read_image(image_file), read_image(batch / local["source_path"]))
    for identifier in ("ambiguous.jpg", "pending.jpg", "non-target.jpg"):
        assert manifest[identifier]["exclusion_reason"]
        assert manifest[identifier]["image_file"] == ""


def test_unsure_background_confirmation_only_applies_to_selected_species(snapshot, tmp_path):
    directory, batch, data = snapshot
    confirmed = data["images"][3]
    unsure = dict(active=True, decision="Unsure / mixed", box=[22, 2, 28, 10])
    pending = dict(active=True, decision="pending")
    data["images"].extend([
        dict(confirmed, image_id="confirmed.jpg", crops=confirmed["crops"] + [unsure]),
        dict(confirmed, image_id="pending-decision.jpg", crops=confirmed["crops"] + [pending]),
        dict(data["images"][0], image_id="unconfirmed.jpg", crops=data["images"][0]["crops"] + [unsure]),
    ])
    _save(directory, data)
    output = tmp_path / "dataset"
    export_dataset(directory, output, batch_dir=batch, unsure_background_for=("n_fruticosa",))
    manifest = _manifest(output)
    row = manifest["confirmed.jpg"]
    label_file = output / "labels" / row["split"] / (Path(row["image_file"]).stem + ".txt")
    assert label_file.read_text(encoding="utf-8") == "1 0.3 0.35 0.4 0.5\n"
    assert manifest["pending-decision.jpg"]["exclusion_reason"]
    assert manifest["unconfirmed.jpg"]["exclusion_reason"]


def test_split_is_deterministic_and_preserves_parent_and_duplicate_groups(snapshot, tmp_path):
    directory, batch, data = snapshot
    first = data["images"][0]
    second = data["images"][1]
    second["parent_image_id"] = first["parent_image_id"]
    data["images"].append(dict(first, image_id="alias.jpg", parent_image_id="alias.jpg"))
    source = Path(data["root"]) / "extra.jpg"
    Image.new("RGB", (40, 20), (150, 100, 30)).save(source)
    data["images"].append(dict(first, image_id="extra.jpg", parent_image_id="extra.jpg",
                               source_path="extra.jpg", batch_local_source=False,
                               sha256=sha256(source.read_bytes()).hexdigest()))
    _save(directory, data)
    output = tmp_path / "first"
    export_dataset(directory, output, batch_dir=batch)
    manifest = _manifest(output)
    assert manifest[first["image_id"]]["split"] == manifest[second["image_id"]]["split"] == manifest["alias.jpg"]["split"]
    for split in ("train", "val"):
        assert {row["target"] for row in manifest.values() if row["split"] == split} == set(TARGET_CLASSES)
    data["images"].reverse()
    _save(directory, data)
    repeat = tmp_path / "second"
    export_dataset(directory, repeat, batch_dir=batch)
    assert _manifest(repeat) == manifest


@pytest.mark.parametrize("failure,pattern", [
    ("box", "Invalid bounding box"), ("hash", "Source image changed"),
    ("missing", "Cannot read original"), ("batch", "Pass --batch-dir"),
    ("exists", "already exists"), ("evaluation", "Crop Review training snapshot"),
])
def test_failed_export_leaves_no_partial_dataset(snapshot, tmp_path, failure, pattern):
    directory, batch, data = snapshot
    output = tmp_path / "dataset"
    row = data["images"][0]
    if failure == "box":
        row["crops"][0]["box"] = [0, 0, 41, 20]
    elif failure == "hash":
        row["sha256"] = "a" * 64
    elif failure == "missing":
        (batch / row["source_path"]).unlink()
    elif failure == "batch":
        batch = None
    elif failure == "exists":
        output.mkdir()
        (output / "keep.txt").write_text("keep", encoding="utf-8")
    elif failure == "evaluation":
        data["collection"] = "evaluation"
    _save(directory, data)
    with pytest.raises((ValueError, FileExistsError), match=pattern):
        export_dataset(directory, output, batch_dir=batch)
    if failure == "exists":
        assert list(output.iterdir()) == [output / "keep.txt"]
    else:
        assert not output.exists()
    assert list(tmp_path.glob(".yolo-export-*")) == []
