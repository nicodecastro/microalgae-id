"""Verified parent identity and leakage-safe, precomputed splits."""
import csv

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from sklearn.base import clone


def validate_identity(image_ids, parent_ids, count):
    arrays = []
    for name, values in (("image_id", image_ids), ("parent_image_id", parent_ids)):
        values = np.asarray(values)
        if values.ndim != 1 or len(values) != count or any(
            not isinstance(v, str) or not v.strip() for v in values
        ):
            raise ValueError(f"{name} requires one nonblank string per image")
        arrays.append(values.copy())
    if len(set(arrays[0])) != count:
        raise ValueError("image_id values must be unique")
    return tuple(arrays)


def load_parent_groups(manifest, path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)) or set(header) != {"image_id", "parent_image_id"}:
        raise ValueError("Parent CSV requires image_id,parent_image_id columns")
    mapping = pd.read_csv(path, keep_default_na=False, dtype=str)
    validate_identity(mapping.image_id, mapping.parent_image_id, len(mapping))
    if set(mapping.image_id) != set(manifest.image_id):
        raise ValueError("Parent mapping has missing or unexpected image IDs")
    result = manifest.copy()
    result["parent_image_id"] = result.image_id.map(mapping.set_index("image_id").parent_image_id)
    return result


def grouped_folds(labels, parents, count, seed, context):
    y, groups = np.asarray(labels), np.asarray(parents)
    classes = set(y)
    if len(classes) < 2:
        raise ValueError(f"{context} requires at least two classes")
    if any(len(set(groups[y == label])) < count for label in classes):
        raise ValueError(f"{context} needs at least {count} parent groups per class")
    folds = list(StratifiedGroupKFold(count, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, groups))
    for train, test in folds:
        if (set(groups[train]) & set(groups[test]) or
                set(y[train]) != classes or set(y[test]) != classes):
            raise ValueError(f"{context} cannot preserve parent isolation and class coverage")
    return folds


def legacy_splits(frame, labels, estimator, grid):
    if not {"image_id", "parent_image_id"}.issubset(frame):
        raise ValueError("Training requires image_id and parent_image_id")
    _, parents = validate_identity(frame.image_id, frame.parent_image_id, len(labels))
    train, test = grouped_folds(labels, parents, 5, 42, "Holdout")[0]
    inner = grouped_folds(labels[train], parents[train], 10, 42, "Holdout tuning")
    smallest = min(len(indices) for indices, _ in inner)
    for parameters in ParameterGrid(grid):
        candidate = clone(estimator).set_params(**parameters)
        candidate._validate_params()
        if hasattr(candidate, "n_neighbors") and candidate.n_neighbors > smallest:
            raise ValueError("KNN n_neighbors exceeds the smallest CV training set")
    return train, test, inner
