"""Export reviewed full microscope fields for Ultralytics object detection."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import cv2
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import yaml

from microalgae.imaging.crop_review import TARGET_CLASSES, write_crop
from scripts._detection_references import source_path
from scripts.run_detection_pilot import source_groups


def _accepted_boxes(row, unsure_background_for):
    if row["status"] == "failed":
        return [], "source review failed"
    if row["reviewed"] is not True:
        return [], "field not reviewed"
    active = [crop for crop in row["crops"] if crop["active"]]
    excluded_decisions = {"pending"}
    if row["target"] not in unsure_background_for:
        excluded_decisions.add("Unsure / mixed")
    if any(crop["decision"] in excluded_decisions for crop in active):
        return [], "field has an unresolved or ambiguous crop"
    boxes = [crop["box"] for crop in active if crop["decision"] == "Accept target"]
    if not boxes:
        return [], "field has no accepted target"
    if row["target"] not in TARGET_CLASSES:
        raise ValueError(f"Unsupported target species for {row['image_id']}: {row['target']!r}")
    return boxes, ""


def _split(rows):
    sources = {
        row["image_id"]: {"parent_image_id": row["parent_image_id"], "specimen_id": ""}
        for row in rows
    }
    groups = [0] * len(rows)
    for group in source_groups(rows, sources):
        group_id = min(group)
        for index in group:
            groups[index] = group_id
    labels = [row["target"] for row in rows]
    required = set(TARGET_CLASSES)
    groups_per_species = {species: set() for species in TARGET_CLASSES}
    for species, group in zip(labels, groups):
        groups_per_species[species].add(group)
    folds = min(5, min(map(len, groups_per_species.values())))
    if folds < 2:
        raise ValueError("Train and validation need at least two independent source groups per species")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    for train, validation in splitter.split(np.zeros(len(rows)), labels, groups):
        if {labels[i] for i in train} == {labels[i] for i in validation} == required:
            assignments = {rows[i]["image_id"]: "train" for i in train}
            assignments.update({rows[i]["image_id"]: "val" for i in validation})
            return assignments
    raise ValueError("No grouped split contains all three target species in train and validation")


def _export_field(staging, row, boxes, root, image_file, label_file):
    identifier = row["image_id"]
    relative = row.get("source_path", identifier)
    path = source_path(root, relative)
    try:
        original = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read original for {identifier}: {path}") from error
    if sha256(original).hexdigest() != row["sha256"]:
        raise ValueError(f"Source image changed: {identifier}")
    image = cv2.imdecode(np.frombuffer(original, dtype=np.uint8),
                         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Cannot decode original for {identifier}: {path}")
    height, width = image.shape[:2]
    class_id = TARGET_CLASSES.index(row["target"])
    labels = []
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
            raise ValueError(f"Invalid bounding box for {identifier}: {box!r}")
        left, top, right, bottom = box
        normalized = ((left + right) / (2 * width), (top + bottom) / (2 * height),
                      (right - left) / width, (bottom - top) / height)
        labels.append(f"{class_id} " + " ".join(f"{value:.12g}" for value in normalized))
    write_crop(staging / image_file, image, (0, 0, width, height))
    (staging / label_file).write_text("\n".join(labels) + "\n", encoding="utf-8")


def export_dataset(snapshot_dir, output_dir, *, batch_dir=None, unsure_background_for=()):
    snapshot_dir = Path(snapshot_dir).resolve()
    output = Path(output_dir).resolve()
    batch_dir = Path(batch_dir).resolve() if batch_dir is not None else None
    snapshot = json.loads((snapshot_dir / "review_snapshot.json").read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1 or snapshot.get("collection") != "training":
        raise ValueError("Expected a version 1 Crop Review training snapshot")
    root = Path(snapshot["root"]).resolve()
    roots = (snapshot_dir, root) if batch_dir is None else (snapshot_dir, root, batch_dir)
    if any(output.is_relative_to(path) for path in roots):
        raise ValueError("Output must be outside the snapshot and source directories")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    rows = sorted(snapshot["images"], key=lambda row: row["image_id"])
    selections = {}
    for row in rows:
        identifier = row["image_id"]
        if identifier in selections:
            raise ValueError(f"Duplicate source image ID: {identifier}")
        boxes, reason = _accepted_boxes(row, unsure_background_for)
        if boxes and (not isinstance(row["parent_image_id"], str) or not row["parent_image_id"].strip()):
            raise ValueError(f"Missing parent image ID for {identifier}")
        if boxes and row.get("batch_local_source") and batch_dir is None:
            raise ValueError(f"Pass --batch-dir to resolve the original for {identifier}")
        selections[identifier] = (boxes, reason)
    splits = _split([row for row in rows if selections[row["image_id"]][0]])

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".yolo-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        for split in ("train", "val"):
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
        config = dict(path=output.as_posix(), train="images/train", val="images/val",
                      names=dict(enumerate(TARGET_CLASSES)))
        (staging / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        with (staging / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "parent_image_id", "source_sha256", "pilot_partition", "split",
                             "image_file", "target", "box_count", "exclusion_reason"])
            for index, row in enumerate(rows, start=1):
                identifier = row["image_id"]
                boxes, reason = selections[identifier]
                split = splits.get(identifier, "")
                image_file = ""
                if split:
                    image_file = f"images/{split}/{index:06d}.png"
                    label_file = f"labels/{split}/{index:06d}.txt"
                    source_root = batch_dir if row.get("batch_local_source") else root
                    _export_field(staging, row, boxes, source_root, image_file, label_file)
                writer.writerow([identifier, row.get("parent_image_id"), row["sha256"],
                                 row.get("pilot_partition"), split, image_file, row["target"], len(boxes), reason])
        if output.exists():
            raise FileExistsError(f"Output directory already exists: {output}")
        staging.rename(output)
    return Counter(splits.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-dir", type=Path, help="Crop Review batch holding imported originals")
    parser.add_argument("--unsure-background-for", choices=TARGET_CLASSES, action="append", default=[],
                        help="Treat Unsure / mixed crops as background for a confirmed target species")
    args = parser.parse_args()
    try:
        counts = export_dataset(args.snapshot_dir, args.output_dir, batch_dir=args.batch_dir,
                                unsure_background_for=args.unsure_background_for)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, cv2.error) as error:
        parser.exit(1, f"YOLO dataset export failed: {error}\n")
    print(f"Exported {counts['train']} training and {counts['val']} validation fields to {args.output_dir}")


if __name__ == "__main__":
    main()
