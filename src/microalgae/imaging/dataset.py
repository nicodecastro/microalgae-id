"""Portable image identity and per-extraction failure reports."""

from contextlib import contextmanager
import csv
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FAILURE_COLUMNS = ["image_id", "Class", "error_type", "reason"]


def image_id(image_path, dataset_dir):
    """Identify an image by its path beneath the dataset root, not its row."""
    return Path(image_path).resolve().relative_to(Path(dataset_dir).resolve()).as_posix()


def build_dataset_manifest(dataset_dir, class_subfolders):
    """List supported files in class order, then filename order.

    Parent identity is intentionally unknown. This is a path inventory, not
    proof of readable images, correct labels, or independent source groups.
    """
    rows = []
    classes = list(class_subfolders)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Select at least one class, without duplicate class names")
    for label in classes:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Class names must be nonblank strings")
        class_dir = Path(dataset_dir) / label
        # Reject class directories outside the root, including escaping links.
        image_id(class_dir, dataset_dir)
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append([image_id(path, dataset_dir), label, ""])
    if not rows:
        raise ValueError("No supported images found for the dataset manifest")
    return pd.DataFrame(rows, columns=["image_id", "Class", "parent_image_id"])


def failure_report_path(feature_path):
    path = Path(feature_path)
    return path.with_name(f"{path.stem}_failures.csv")


class ExtractionFailureReport:
    """Reset a sibling ledger, and persist failures before a skip or exception.

    Successful attempts leave a header-only ledger. A failed retry leaves the
    old feature table untouched, but its nonempty ledger blocks strict merging.
    """

    def __init__(self, feature_path, dataset_dir):
        self.path = failure_report_path(feature_path)
        self.dataset_dir = dataset_dir
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(FAILURE_COLUMNS)

    def record(self, path, label, error):
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([
                image_id(path, self.dataset_dir), label,
                type(error).__name__, str(error),
            ])

    @contextmanager
    def track(self, path, label):
        """Keep the caller's exception type and skip/raise policy unchanged."""
        try:
            yield
        except Exception as error:
            self.record(path, label, error)
            raise
