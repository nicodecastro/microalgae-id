"""Shared dataset traversal, identity, failure accounting, and table export.

Family adapters calculate one image at a time. They retain their preprocessing,
descriptor parameters, previews, and exception context; this module owns the
dataset-level lifecycle used by every feature family.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..dataset import ExtractionFailureReport, IMAGE_EXTENSIONS, image_id


@dataclass(frozen=True)
class SkippedImage:
    """An expected unusable image that must be recorded and warned about."""

    reason: str


def export_feature_table(
    dataset_dir: str | Path,
    class_subfolders: Iterable[str],
    output_path: str | Path,
    feature_names: Sequence[str],
    extract_image: Callable[[Path, str], Iterable[float] | SkippedImage],
    *,
    empty_message: str,
    standardize: bool = False,
    sort_paths: bool = True,
    require_class_images: bool = False,
) -> pd.DataFrame:
    """Export one family, preserving its ordering and empty-class policy.

    The adapter returns raw values or an explicit skip. Exceptions propagate
    unchanged after their image is recorded. Each attempt resets the sibling
    failure ledger; no table is written until every image has been attempted
    successfully or explicitly skipped. Scaling excludes identity and labels.
    """
    output_path = Path(output_path)
    failures = ExtractionFailureReport(output_path, dataset_dir)
    rows = []
    for label in class_subfolders:
        class_dir = Path(dataset_dir) / label
        paths = [
            class_dir / name for name in os.listdir(class_dir)
            if (class_dir / name).is_file() and Path(name).suffix.lower() in IMAGE_EXTENSIONS
        ]
        if sort_paths:
            paths.sort()
        if require_class_images and not paths:
            raise ValueError(f"No supported images found in {class_dir}")
        print(f"Class: {label} ({len(paths)} images)")
        for path in paths:
            with failures.track(path, label):
                values = extract_image(path, label)
                if isinstance(values, SkippedImage):
                    failures.record(path, label, ValueError(values.reason))
                    warnings.warn(values.reason, stacklevel=3)
                    continue
                rows.append([*values, image_id(path, dataset_dir), label])

    if not rows:
        raise ValueError(empty_message)
    columns = list(feature_names)
    frame = pd.DataFrame(rows, columns=[*columns, "image_id", "Class"])
    if standardize:
        frame[columns] = StandardScaler().fit_transform(frame[columns])
    frame.to_csv(output_path, index=False)
    return frame
