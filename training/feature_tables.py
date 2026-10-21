"""Strict, keyed feature-table validation and classifier input selection."""

from collections.abc import Mapping
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from microalgae.imaging.dataset import FAILURE_COLUMNS, failure_report_path


FEATURE_FILENAMES = {
    "efd": "EFD_features.csv",
    "gabor": "Gabor_features.csv",
    "geom": "Geometric_shape_features.csv",
    "glcm": "GLCM_features.csv",
    "hu": "Hu_moments_nonminmax_features.csv",
    "lbp": "LBP_features.csv",
    "pc": "Phase_congruency_features.csv",
}
METADATA_COLUMNS = {"image_id", "Class", "parent_image_id"}
ISSUE_COLUMNS = ["feature_block", "image_id", "issue", "detail"]


def _blank(series):
    return series.isna() | series.map(
        lambda value: isinstance(value, str) and not value.strip()
    ).astype(bool)


def _read_table(source):
    if isinstance(source, pd.DataFrame):
        return source.copy()
    # pandas mangles duplicate CSV headers; reject them before parsing.
    with Path(source).open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names")
    # Keep literal identifiers such as 'NA'; only empty fields are missing.
    return pd.read_csv(source, keep_default_na=False, na_values=[""])


def _numeric_issues(frame, columns):
    issues = []
    if not columns:
        issues.append(("no_features", "No numeric feature columns"))
    for column in columns:
        values = frame[column].to_numpy()
        if values.dtype.kind not in "biuf":
            issues.append(("nonnumeric_features", f"Column {column!r} must contain real numeric values"))
        elif not np.isfinite(values).all():
            issues.append(("nonfinite_features", f"Column {column!r} contains missing or non-finite values"))
    return issues


def _audit(manifest, feature_tables):
    issues = []
    frames = {}

    def add(block, issue, detail, identifier=""):
        issues.append([block, identifier, issue, detail])

    def load(source, block):
        try:
            frame = _read_table(source)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            add(block, "unreadable_table", str(error))
            return None
        if frame.columns.has_duplicates:
            add(block, "duplicate_columns", "Column names must be unique")
            return None
        return frame

    def identity(frame, block):
        for name in ("image_id", "Class"):
            if name not in frame:
                add(block, "missing_column", f"Required column {name!r} is missing")
                continue
            blank = _blank(frame[name])
            if blank.any():
                add(block, f"blank_{name}", f"{int(blank.sum())} rows have missing/blank {name}")
        if "image_id" in frame:
            valid = frame["image_id"].map(lambda value: isinstance(value, str) and bool(value.strip())).astype(bool)
            if (~valid & ~_blank(frame["image_id"])).any():
                add(block, "invalid_image_id", "Image IDs must be strings")
            for identifier in frame.loc[valid & frame["image_id"].duplicated(keep=False), "image_id"].unique():
                add(block, "duplicate_image_id", "Image ID appears more than once", identifier)
        if frame.empty:
            add(block, "empty_table", "Table has no image rows")

    manifest_frame = load(manifest, "manifest")
    if manifest_frame is not None:
        identity(manifest_frame, "manifest")
    manifest_ok = not issues
    if manifest_ok:
        expected = manifest_frame.set_index("image_id")["Class"]
    else:
        expected = None

    if not isinstance(feature_tables, Mapping) or not feature_tables:
        add("selection", "empty_selection", "Provide an explicit nonempty mapping of feature families to tables")
        feature_tables = {}
    output_names = set()
    for block, source in feature_tables.items():
        if not isinstance(block, str) or not block.strip():
            add(str(block), "invalid_family", "Feature family names must be nonblank strings")
            continue
        frame = load(source, block)
        if frame is not None:
            frames[block] = frame
            identity(frame, block)
            columns = [name for name in frame if name not in METADATA_COLUMNS]
            for issue, detail in _numeric_issues(frame, columns):
                add(block, issue, detail)
            for column in columns:
                name = f"{block}__{column}"
                if name in output_names:
                    add(block, "column_collision", f"Combined column {name!r} is not unique")
                output_names.add(name)
            if expected is not None and "image_id" in frame:
                valid = frame["image_id"].map(lambda value: isinstance(value, str) and bool(value.strip())).astype(bool)
                actual_ids = set(frame.loc[valid, "image_id"])
                for identifier in sorted(set(expected.index) - actual_ids):
                    add(block, "missing_image", "Manifest image has no feature row", identifier)
                for identifier in sorted(actual_ids - set(expected.index)):
                    add(block, "unexpected_image", "Feature image is absent from manifest", identifier)
                if "Class" in frame:
                    for identifier, label in frame.loc[valid, ["image_id", "Class"]].itertuples(index=False, name=None):
                        if identifier in expected and (pd.isna(label) or label != expected[identifier]):
                            add(block, "label_conflict", f"Expected Class {expected[identifier]!r}; got {label!r}", identifier)
        if not isinstance(source, pd.DataFrame):
            ledger_path = failure_report_path(source)
            if ledger_path.exists():
                ledger = load(ledger_path, block)
                if ledger is not None:
                    if not set(FAILURE_COLUMNS).issubset(ledger):
                        add(block, "invalid_failure_report", f"Malformed failure report: {ledger_path}")
                    elif not ledger.empty:
                        for identifier, label, error_type, reason in ledger[FAILURE_COLUMNS].itertuples(index=False, name=None):
                            add(block, "extraction_failure", f"{label}: {error_type}: {reason}", identifier)
    return manifest_frame, frames, pd.DataFrame(issues, columns=ISSUE_COLUMNS)


def audit_feature_tables(manifest, feature_tables):
    """Return every detected issue; paths and in-memory DataFrames are accepted.

    Path inputs also check any sibling failure ledger. Legacy ID-less tables
    are rejected even if their row counts happen to match the manifest.
    """
    return _audit(manifest, feature_tables)[2]


def combine_feature_tables(manifest, feature_tables, output_path=None):
    """Join an explicit selection by ID, or raise before writing any output."""
    manifest, frames, issues = _audit(manifest, feature_tables)
    if not issues.empty:
        raise ValueError("Feature alignment failed:\n" + issues.to_string(index=False))
    identifiers = manifest["image_id"].tolist()
    blocks = []
    for family, frame in frames.items():
        columns = [name for name in frame if name not in METADATA_COLUMNS]
        block = frame.set_index("image_id").loc[identifiers, columns].reset_index(drop=True)
        blocks.append(block.add_prefix(f"{family}__"))
    combined = pd.concat(blocks, axis=1)
    combined["image_id"] = identifiers
    combined["Class"] = manifest["Class"].to_numpy()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
    return combined


def classifier_inputs(frame):
    """Return numeric X and labels y, accepting keyed and legacy tables.

    Drop only known metadata, never silently drop an unexpected text feature.
    This does not fit scaling or change training/evaluation policy.
    """
    if frame.columns.has_duplicates:
        raise ValueError("Classifier column names must be unique")
    if "Class" not in frame or frame.empty or _blank(frame["Class"]).any():
        raise ValueError("Classifier input requires nonblank Class labels and at least one row")
    columns = [name for name in frame if name not in METADATA_COLUMNS]
    issues = _numeric_issues(frame, columns)
    if issues:
        raise ValueError("; ".join(detail for _, detail in issues))
    return frame[columns].to_numpy(), frame["Class"].to_numpy()
