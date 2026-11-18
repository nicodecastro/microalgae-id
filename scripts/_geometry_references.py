"""Identity-checked reference-mask data shared by annotation and scoring commands."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from scripts.diagnose_geometry import _project_path, _reference_canvas, _save_png, baseline_preprocessing, HYSTERESIS_EXPERIMENT_PROFILES
from microalgae.imaging.dataset import image_id


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(root, identifier):
    """Resolve canonical IDs without permitting traversal or symlink escape."""
    root = Path(root).resolve()
    path = (root / identifier).resolve()
    if image_id(path, root) != identifier:
        raise ValueError(f"Noncanonical image ID: {identifier}")
    return path


def mask_path(root, identifier):
    return child(Path(root) / "masks", identifier + "_mask.png")


def load_comparison(directory):
    directory = _project_path(directory)
    settings_path = directory / "resolved_settings.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    selection = pd.read_csv(directory / "selection.csv", keep_default_na=False)
    if (not {"image_id", "Class"}.issubset(selection) or selection.empty
            or selection.image_id.duplicated().any()):
        raise ValueError("Invalid comparison selection")
    sources = json.loads((directory / "source_images.json").read_text(encoding="utf-8"))
    inventory = {row["image_id"]: row for row in sources}
    if len(inventory) != len(sources) or set(inventory) != set(selection.image_id):
        raise ValueError("Comparison source inventory does not match selection")
    root = Path(settings["dataset_dir"]).resolve()
    for identifier in selection.image_id:
        child(root, identifier)
    baseline_preprocessing(settings)
    if settings.get("experiment") == "hysteresis_refinement":
        if set(settings["profiles"]) != set(HYSTERESIS_EXPERIMENT_PROFILES):
            raise ValueError("Hysteresis experiment requires exactly five profiles")
    elif not {"canny_baseline", "otsu_mask"}.issubset(settings["profiles"]):
        raise ValueError("Comparison requires Canny and dark Otsu profiles")
    return directory, settings, selection, inventory, digest(settings_path)


def load_canvas(settings, source):
    path = child(Path(settings["dataset_dir"]), source["image_id"])
    if digest(path) != source["sha256"]:
        raise ValueError(f"Source image hash changed: {source['image_id']}")
    raw = cv2.imread(str(path))
    if raw is None:
        raise ValueError(f"Unreadable image: {path}")
    canvas, content = _reference_canvas(raw, baseline_preprocessing(settings))
    if list(content.shape) != source["canvas_shape"]:
        raise ValueError(f"Canvas shape changed: {source['image_id']}")
    return canvas, content


def rasterize(polygons, shape):
    """Apply additive/subtractive polygons in drawing order, at canvas pixels."""
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon["vertices"], dtype=float)
        if (polygon["mode"] not in ("add", "subtract") or points.ndim != 2
                or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all()
                or np.any(points < 0) or np.any(points[:, 0] > shape[1] - 1)
                or np.any(points[:, 1] > shape[0] - 1)):
            raise ValueError("Invalid reference polygon coordinates or mode")
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1 if polygon["mode"] == "add" else 0)
    return mask.astype(bool)


def validate_mask(mask, content, *, approved=True):
    if (mask.shape != content.shape or mask.dtype != bool or np.any(mask & ~content)
            or (approved and not mask.any())):
        raise ValueError("Reference mask must match canvas, exclude padding, and be nonempty when approved")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class ReferenceSession:
    """Persist explicit annotation decisions; never infer a foreground mask."""

    def __init__(self, comparison_dir, output_dir, *, resume=False, select_all=False,
                 import_references=None):
        if resume and (select_all or import_references is not None):
            raise ValueError("Selection and import options apply only when creating a session")
        self.directory, self.settings, selection, inventory, settings_hash = load_comparison(comparison_dir)
        self.selection = selection
        self.output = _project_path(output_dir)
        self.path = self.output / "annotations.json"
        raw = Path(self.settings["dataset_dir"]).resolve()
        source_run = Path(self.settings["source_run"]).resolve()
        if any(self.output.is_relative_to(path) for path in (raw, source_run, self.directory)):
            raise ValueError("Reference output must be outside raw data, source run, and comparison")
        if self.output.exists():
            if not resume:
                raise FileExistsError("Reference output exists; use --resume to continue")
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            mode = self.data.get("selection_mode", "csv")
        else:
            if resume:
                raise FileNotFoundError("Cannot resume a reference directory that does not exist")
            mode = "all" if select_all else "csv"
        if mode not in ("all", "csv"):
            raise ValueError("Invalid reference selection mode")
        if mode == "all":
            identifiers = set(selection.image_id)
        else:
            choices = pd.read_csv(self.directory / "reference_selection.csv", dtype=str, keep_default_na=False)
            if (not {"image_id", "selected"}.issubset(choices) or choices.image_id.duplicated().any()
                    or set(choices.image_id) != set(selection.image_id)):
                raise ValueError("Reference selection IDs must exactly match the comparison")
            flags = choices.selected.str.strip().str.lower()
            if not flags.isin(["true", "false"]).all():
                raise ValueError("reference_selection.selected must contain true or false")
            identifiers = set(choices.loc[flags == "true", "image_id"])
        if not identifiers:
            raise ValueError("Select at least one reference image")
        images = [dict(inventory[identifier], status="pending", polygons=[])
                  for identifier in selection.image_id if identifier in identifiers]
        expected = {"comparison_dir": str(self.directory), "settings_sha256": settings_hash,
                    "resize": baseline_preprocessing(self.settings)["resize"],
                    "selection_mode": mode, "images": images}
        for source in images:
            load_canvas(self.settings, source)
        if resume:
            self._check_identity(expected)
            self.validate_saved()
        else:
            imported_masks = []
            if import_references is not None:
                imported_masks = self._import_references(expected, import_references)
            self.output.mkdir(parents=True, exist_ok=False)
            for identifier, contents in imported_masks:
                destination = mask_path(self.output, identifier)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(contents)
            self.data = expected
            write_json(self.path, self.data)

    def _import_references(self, expected, reference_dir):
        """Validate the entire import before creating any destination files."""
        directory = _project_path(reference_dir)
        if self.output.is_relative_to(directory):
            raise ValueError("Imported reference output must be outside the source reference directory")
        source_path = directory / "annotations.json"
        metadata = json.loads(source_path.read_text(encoding="utf-8"))
        source = ReferenceSession(metadata["comparison_dir"], directory, resume=True)
        if (baseline_preprocessing(source.settings) != baseline_preprocessing(self.settings)
                or source.data["resize"] != expected["resize"]):
            raise ValueError("Imported references require identical baseline preprocessing and resizing")
        targets = {row["image_id"]: row for row in expected["images"]}
        labels = self.selection.set_index("image_id")["Class"]
        source_labels = source.selection.set_index("image_id")["Class"]
        masks = []
        for row in source.data["images"]:
            identifier = row["image_id"]
            if identifier not in targets:
                raise ValueError(f"Imported reference ID is outside the destination selection: {identifier}")
            target = targets[identifier]
            if (labels[identifier] != source_labels[identifier]
                    or any(target[key] != row[key] for key in ("sha256", "canvas_shape"))):
                raise ValueError(f"Imported reference identity/label mismatch: {identifier}")
            target.update(status=row["status"], polygons=json.loads(json.dumps(row["polygons"])))
            path = mask_path(source.output, identifier)
            if path.exists():
                masks.append((identifier, path.read_bytes()))
        expected["imported_from"] = {"directory": str(source.output), "annotations_sha256": digest(source.path)}
        return masks

    def _check_identity(self, expected):
        for key in ("comparison_dir", "settings_sha256", "resize"):
            if self.data.get(key) != expected[key]:
                raise ValueError(f"Reference provenance mismatch: {key}")
        actual = [{k: row[k] for k in ("image_id", "sha256", "canvas_shape")} for row in self.data["images"]]
        wanted = [{k: row[k] for k in ("image_id", "sha256", "canvas_shape")} for row in expected["images"]]
        if actual != wanted:
            raise ValueError("Reference IDs, source hashes, or canvas settings changed")

    def validate_saved(self):
        for row in self.data["images"]:
            if row["status"] not in ("pending", "approved", "uncertain"):
                raise ValueError("Invalid annotation status")
            _, content = load_canvas(self.settings, row)
            mask = rasterize(row["polygons"], content.shape)
            validate_mask(mask, content, approved=row["status"] == "approved")
            path = mask_path(self.output, row["image_id"])
            if row["status"] == "approved" or path.exists():
                saved = read_binary(path, content.shape)
                if not np.array_equal(saved, mask):
                    raise ValueError(f"Saved mask differs from polygons: {row['image_id']}")

    def save(self, index, polygons, status):
        if status not in ("approved", "uncertain"):
            raise ValueError("Save status must be approved or uncertain")
        row = self.data["images"][index]
        _, content = load_canvas(self.settings, row)
        mask = rasterize(polygons, content.shape)
        validate_mask(mask, content, approved=status == "approved")
        path = mask_path(self.output, row["image_id"])
        temporary = path.with_name(path.stem + ".tmp.png")
        previous = path.read_bytes() if path.exists() else None
        _save_png(temporary, mask.astype(np.uint8) * 255)
        temporary.replace(path)
        # Store a copy: subsequent drawing must not mutate the saved decision.
        updated = json.loads(json.dumps(self.data))
        updated["images"][index].update(status=status, polygons=json.loads(json.dumps(polygons)))
        try:
            write_json(self.path, updated)
        except OSError:
            if previous is None:
                path.unlink()
            else:
                temporary.write_bytes(previous)
                temporary.replace(path)
            raise
        self.data = updated


def read_binary(path, shape):
    # Avoid OpenCV's native missing-file diagnostic on Windows; keep the same error.
    if not Path(path).is_file():
        raise ValueError(f"Missing, nonbinary, or mismatched mask: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (image is None or image.ndim != 2 or image.dtype != np.uint8
            or image.shape != tuple(shape) or not np.isin(image, [0, 255]).all()):
        raise ValueError(f"Missing, nonbinary, or mismatched mask: {path}")
    return image != 0
