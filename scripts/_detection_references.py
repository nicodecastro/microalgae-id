"""Original-pixel bounding-box references, independent of segmentation predictions."""

from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import random

import cv2

from microalgae.imaging.dataset import IMAGE_EXTENSIONS


SPECIES = ("unknown", "m_aeruginosa", "n_fruticosa", "n_palea", "other_alga")
KINDS = ("cell", "colony_group", "unresolved_group")
STATUSES = ("pending", "reviewed", "uncertain")
PARTITIONS = ("unassigned", "development", "evaluation")
POLICY = "All identifiable algal cells or intact colonies/groups; exclude debris, rulers and text."


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_image(path):
    # Keep file pixel coordinates independent of EXIF display rotation.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def source_path(root, identifier):
    if not isinstance(identifier, str) or not identifier or "\\" in identifier:
        raise ValueError("image_id must be a root-relative path using forward slashes")
    path = (root / identifier).resolve()
    if not path.is_relative_to(root) or Path(identifier).is_absolute():
        raise ValueError(f"Image escapes dataset root: {identifier}")
    if path.relative_to(root).as_posix() != identifier:
        raise ValueError("image_id must use its canonical root-relative path")
    return path


def validate_image(row):
    width, height = row["width"], row["height"]
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("Image dimensions must be positive integers")
    if row["status"] not in STATUSES or row["partition"] not in PARTITIONS:
        raise ValueError("Invalid review status or partition")
    if not isinstance(row["notes"], str) or type(row["empty_confirmed"]) is not bool:
        raise ValueError("Invalid image notes or empty confirmation")
    if not isinstance(row["objects"], list):
        raise ValueError("Objects must be a list")
    identifiers = set()
    for obj in row["objects"]:
        identifier = obj["object_id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Object IDs must be nonblank and unique within an image")
        identifiers.add(identifier)
        box = obj["box"]
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) is not int for v in box)
                or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
            raise ValueError("Box must be integer [left, top, right, bottom) inside the image")
        if obj["species"] not in SPECIES or obj["kind"] not in KINDS:
            raise ValueError("Invalid species or object kind")
        if any(type(obj[k]) is not bool for k in ("uncertain", "truncated")) or not isinstance(obj["notes"], str):
            raise ValueError("Invalid object flags or notes")
        if obj["kind"] == "unresolved_group" and not obj["uncertain"]:
            raise ValueError("Unresolved groups must be flagged uncertain")
    if row["objects"] and row["empty_confirmed"]:
        raise ValueError("An image with boxes cannot be confirmed empty")
    if row["status"] == "reviewed" and not row["objects"] and not row["empty_confirmed"]:
        raise ValueError("Confirm 'No algae visible' before reviewing an empty image")


class DetectionSession:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.path = self.directory / "annotations.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if self.data.get("schema_version") != 1 or self.data.get("policy") != POLICY:
            raise ValueError("Unsupported detection reference session")
        self.root = Path(self.data["dataset_dir"]).resolve()
        identifiers = set()
        paths = set()
        if not self.data["images"]:
            raise ValueError("Reference session needs images")
        for row in self.data["images"]:
            if row["image_id"] in identifiers:
                raise ValueError("Duplicate image_id")
            identifiers.add(row["image_id"])
            path = self.verify_source(row)
            key = os.path.normcase(str(path))
            if key in paths:
                raise ValueError("Duplicate source image")
            paths.add(key)
            validate_image(row)
        self._saved_digest = digest(self.path)

    @classmethod
    def create(cls, root, directory, *, per_class=10, seed=42, image_ids=None):
        root, directory = Path(root).resolve(), Path(directory).resolve()
        if directory.exists():
            raise FileExistsError("Session directory exists; use --session to resume")
        if directory.is_relative_to(root):
            raise ValueError("Session must be outside the source dataset")
        if type(per_class) is not int or per_class < 1:
            raise ValueError("per_class must be a positive integer")
        explicit_list = image_ids is not None
        if image_ids is None:
            rng = random.Random(seed)
            image_ids = []
            for folder in sorted(p for p in root.iterdir() if p.is_dir()):
                candidates = sorted(p.relative_to(root).as_posix() for p in folder.iterdir()
                                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
                image_ids.extend(sorted(rng.sample(candidates, min(per_class, len(candidates)))))
        if not image_ids or len(set(image_ids)) != len(image_ids):
            raise ValueError("Select at least one image, without duplicate IDs")
        rows = []
        paths = set()
        for identifier in image_ids:
            path = source_path(root, identifier)
            key = os.path.normcase(str(path))
            if key in paths:
                raise ValueError("Duplicate source image")
            paths.add(key)
            image = read_image(path)
            height, width = image.shape[:2]
            rows.append({"image_id": identifier, "sha256": digest(path), "width": width, "height": height,
                         "source_folder": str(Path(identifier).parent).replace("\\", "/"),
                         "status": "pending", "partition": "unassigned", "notes": "",
                         "empty_confirmed": False, "objects": []})
        data = {"schema_version": 1, "dataset_dir": str(root), "policy": POLICY,
                "coordinate_system": "Unrotated source pixels; [left, top, right, bottom), right/bottom exclusive",
                "selection": {"seed": seed, "per_class": per_class, "explicit_list": explicit_list},
                "images": rows}
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "annotations.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return cls(directory)

    def verify_source(self, row):
        path = source_path(self.root, row["image_id"])
        if digest(path) != row["sha256"]:
            raise ValueError(f"Source image changed: {row['image_id']}")
        if read_image(path).shape[:2] != (row["height"], row["width"]):
            raise ValueError(f"Source dimensions changed: {row['image_id']}")
        return path

    def save(self, index, row):
        current = self.data["images"][index]
        for name in ("image_id", "sha256", "width", "height", "source_folder"):
            if row[name] != current[name]:
                raise ValueError(f"Cannot edit source identity: {name}")
        self.verify_source(row)
        validate_image(row)
        if digest(self.path) != self._saved_digest:
            raise ValueError("Session changed in another window. Reopen before saving.")
        data = deepcopy(self.data)
        data["images"][index] = deepcopy(row)
        payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
        # A failed replacement leaves the previous complete annotation file intact.
        temporary = self.directory / "annotations.tmp"
        temporary.write_text(payload, encoding="utf-8")
        (self.directory / "annotations.previous.json").write_bytes(self.path.read_bytes())
        temporary.replace(self.path)
        self.data = data
        self._saved_digest = digest(self.path)

    def export(self, directory):
        # Preserve all references; uncertain regions remain explicit ignore candidates.
        for row in self.data["images"]:
            self.verify_source(row)
            validate_image(row)
        if digest(self.path) != self._saved_digest:
            raise ValueError("Session changed in another window. Reopen before exporting.")
        directory = Path(directory).resolve()
        if directory.is_relative_to(self.root):
            raise ValueError("Export must be outside the source dataset")
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "annotations.json").write_bytes(self.path.read_bytes())
        with (directory / "images.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["image_id", "sha256", "width", "height", "status", "partition", "empty_confirmed", "object_count", "notes"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.data["images"]:
                writer.writerow({**{k: row[k] for k in fields if k != "object_count"}, "object_count": len(row["objects"])})
        with (directory / "boxes.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["image_id", "parent_image_id", "image_status", "partition", "object_id", "left", "top", "right", "bottom", "species", "kind", "uncertain", "truncated", "ignore", "notes"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.data["images"]:
                for obj in row["objects"]:
                    writer.writerow({"image_id": row["image_id"], "parent_image_id": row["image_id"],
                                     "image_status": row["status"], "partition": row["partition"],
                                     **{k: obj[k] for k in ("object_id", "species", "kind", "uncertain", "truncated", "notes")},
                                     **dict(zip(("left", "top", "right", "bottom"), obj["box"])),
                                     "ignore": row["status"] != "reviewed" or obj["uncertain"]})
        return directory
