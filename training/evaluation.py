"""Shared-split nested evaluation; all learned scaling stays inside pipelines."""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
import json
from numbers import Integral
from pathlib import Path
import platform

import joblib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, get_scorer, precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training.classifiers.knn_model import knn_definition
from training.classifiers.svm_model import svm_definition
from training.feature_tables import METADATA_COLUMNS, classifier_inputs


DEFINITIONS = {"svm": svm_definition, "knn": knn_definition}
DEFAULT_SETTINGS = {
    "models": ["svm", "knn"], "outer_folds": 5, "inner_folds": 5,
    "random_state": 42, "scoring": "f1_macro", "n_jobs": -1,
    "refit_full_data": True,
}


def write_json(path, value):
    """Write strict JSON, including NumPy scalars in metrics and grid results."""
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8")


@dataclass
class EvaluationPlan:
    settings: dict
    specifications: dict
    outer: list
    inner: list
    final: list
    labels: np.ndarray


def prepare_evaluation(experiment, labels):
    """Validate settings/grids and construct every split without fitting models."""
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment must be a mapping")
    allowed = set(DEFAULT_SETTINGS) | {"param_grids", "results_dir"}
    unknown = set(experiment) - allowed
    if unknown:
        raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
    settings = {name: experiment.get(name, default) for name, default in DEFAULT_SETTINGS.items()}
    models = settings["models"]
    if (not isinstance(models, (list, tuple)) or not models
            or any(not isinstance(name, str) or name not in DEFINITIONS for name in models)
            or len(set(models)) != len(models)):
        raise ValueError("experiment.models must select unique names from svm and knn")
    settings["models"] = list(models)
    for name in ("outer_folds", "inner_folds", "random_state", "n_jobs"):
        value = settings[name]
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise ValueError(f"experiment.{name} must be an integer")
        settings[name] = int(value)
    if settings["outer_folds"] < 2 or settings["inner_folds"] < 2:
        raise ValueError("Outer and inner fold counts must be at least 2")
    if not 0 <= settings["random_state"] <= 2**32 - 1:
        raise ValueError("random_state must be between 0 and 2**32 - 1")
    if settings["n_jobs"] == 0:
        raise ValueError("n_jobs cannot be zero")
    if not isinstance(settings["refit_full_data"], bool):
        raise ValueError("refit_full_data must be Boolean")
    if not isinstance(settings["scoring"], str):
        raise ValueError("scoring must be a scikit-learn scorer name")
    get_scorer(settings["scoring"])
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or any(not isinstance(v, str) or not v.strip() for v in y):
        raise ValueError("CV requires nonblank string class labels")
    if len(np.unique(y)) < 2:
        raise ValueError("CV requires at least two classes")

    def split(values, count, context):
        classes, counts = np.unique(values, return_counts=True)
        if counts.min() < count:
            raise ValueError(f"{context} needs at least {count} images per class; got {dict(zip(classes, counts))}")
        splitter = StratifiedKFold(count, shuffle=True, random_state=settings["random_state"])
        return list(splitter.split(np.zeros(len(values)), values))

    outer = split(y, settings["outer_folds"], "Outer CV")
    inner = [split(y[train], settings["inner_folds"], f"Outer fold {fold} inner CV")
             for fold, (train, _) in enumerate(outer, 1)]
    final = split(y, settings["inner_folds"], "Full-data tuning") if settings["refit_full_data"] else []
    smallest_train = min(len(train) for folds in inner + [final] for train, _ in folds)
    overrides = experiment.get("param_grids", {})
    if not isinstance(overrides, Mapping) or set(overrides) - set(models):
        raise ValueError("param_grids must map selected model names to grids")
    specifications = {}
    for name in models:
        estimator, default_grid = DEFINITIONS[name]()
        if "random_state" in estimator.get_params():
            estimator.set_params(random_state=settings["random_state"])
        grid = overrides.get(name, default_grid)
        if not isinstance(grid, Mapping) or not grid:
            raise ValueError(f"{name} grid must be a nonempty parameter mapping")
        grid = dict(grid)
        for parameters in ParameterGrid(grid):
            candidate = clone(estimator).set_params(**parameters)
            # scikit-learn's own constraints catch invalid candidates before fitting.
            candidate._validate_params()
            if name == "knn" and candidate.n_neighbors > smallest_train:
                raise ValueError(f"KNN n_neighbors={candidate.n_neighbors} exceeds the smallest CV training set ({smallest_train}); change the grid explicitly")
        specifications[name] = (estimator, grid)
    return EvaluationPlan(settings, specifications, outer, inner, final, y.copy())


def _split_records(folds, indices, identifiers):
    return [{
        "fold": number,
        "train_indices": indices[train], "test_indices": indices[test],
        "train_ids": identifiers[indices[train]], "test_ids": identifiers[indices[test]],
    } for number, (train, test) in enumerate(folds, 1)]


def save_split_ids(plan, identifiers, path):
    """Persist global row indices and image IDs, shared by all selected models."""
    identifiers = np.asarray(identifiers)
    indices = np.arange(len(identifiers))
    outer = _split_records(plan.outer, indices, identifiers)
    for record, (train, _), inner in zip(outer, plan.outer, plan.inner):
        record["inner"] = _split_records(inner, train, identifiers)
    write_json(path, {
        "index_reference": "Combined_features_keyed.csv, zero-based manifest row order",
        "shared_by_models": plan.settings["models"], "outer": outer,
        "full_data_tuning": _split_records(plan.final, indices, identifiers),
    })


def _scores(y, predicted, classes):
    options = {"average": "macro", "labels": classes, "zero_division": 0}
    return {
        "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "macro_precision": precision_score(y, predicted, **options),
        "macro_recall": recall_score(y, predicted, **options),
        "macro_f1": f1_score(y, predicted, **options),
    }


def _save_confusion(y, predicted, classes, directory):
    matrix = confusion_matrix(y, predicted, labels=classes)
    pd.DataFrame(matrix, index=pd.Index(classes, name="true_class"), columns=classes).to_csv(directory / "confusion_matrix.csv")
    figure = Figure(figsize=(6, 5))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    im = ax.imshow(matrix, cmap="Blues")
    figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes, xlabel="Predicted class", ylabel="True class",
           title="Pooled out-of-fold predictions")
    ax.tick_params(axis="x", labelrotation=45)
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            ax.text(j, i, str(count), ha="center", va="center",
                    color="white" if count > matrix.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(directory / "confusion_matrix.png")


def library_versions():
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib",
                "scikit-image", "opencv-python", "PyYAML", "phasepack", "pyefd")
    return {"python": platform.python_version(), **{name: version(name) for name in packages}}


def evaluate_nested_cv(frame, output_dir, plan, *, progress=None):
    """Evaluate on outer tests, then independently retune/refit on all data.

    ``frame`` must be the strict keyed combination in manifest row order.
    A saved final model is for deployment, not an additional test result.
    """
    X, y = classifier_inputs(frame)
    if "image_id" not in frame or frame.image_id.isna().any() or frame.image_id.duplicated().any():
        raise ValueError("Nested evaluation requires unique nonblank image_id values")
    if not np.array_equal(y, plan.labels):
        raise ValueError("Evaluation rows do not match the planned class order")
    identifiers = frame.image_id.to_numpy()
    if any(not isinstance(v, str) or not v.strip() for v in identifiers):
        raise ValueError("Nested evaluation requires unique nonblank image_id values")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False)
    save_split_ids(plan, identifiers, output_dir / "splits.json")
    features = [name for name in frame if name not in METADATA_COLUMNS]
    classes = list(dict.fromkeys(y))
    versions = library_versions()
    write_json(output_dir / "metadata.json", {
        "feature_columns": features, "class_labels": classes, "library_versions": versions,
        "settings": plan.settings, "standardization": "StandardScaler fitted inside each training fold",
        "source_independence": "Unverified: parent groups are unknown; splits are image-level",
    })

    def search(name, values, labels, splits, directory):
        estimator, grid = plan.specifications[name]
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", clone(estimator))])
        searcher = GridSearchCV(
            pipeline, {f"model__{key}": value for key, value in grid.items()},
            scoring=plan.settings["scoring"], cv=splits, n_jobs=plan.settings["n_jobs"],
            refit=True, error_score="raise", return_train_score=False,
        )
        searcher.fit(values, labels)
        pd.DataFrame(searcher.cv_results_).to_csv(directory / "search_results.csv", index=False)
        selection = {
            "parameters": {key.removeprefix("model__"): value for key, value in searcher.best_params_.items()},
            "inner_selection_score": searcher.best_score_, "scoring": plan.settings["scoring"],
        }
        write_json(directory / "selection.json", selection)
        return searcher, selection

    summaries = {}
    # Complete *all* outer evaluation before fitting either final pipeline.
    for name in plan.settings["models"]:
        directory = output_dir / name
        directory.mkdir()
        predictions = np.empty(len(y), dtype=object)
        coverage = np.zeros(len(y), dtype=int)
        fold_numbers = np.zeros(len(y), dtype=int)
        rows = []
        for fold, ((train, test), inner) in enumerate(zip(plan.outer, plan.inner), 1):
            if progress:
                progress(f"nested_cv/{name}/outer_{fold}")
            fold_dir = directory / f"outer_{fold}"
            fold_dir.mkdir()
            fitted, selection = search(name, X[train], y[train], inner, fold_dir)
            predicted = fitted.predict(X[test])
            predictions[test] = predicted
            coverage[test] += 1
            fold_numbers[test] = fold
            metrics = _scores(y[test], predicted, classes)
            rows.append({"fold": fold, **metrics})
            write_json(fold_dir / "evaluation.json", {"outer_test_metrics": metrics, **selection})
            # Retain completed folds even if a later search fails.
            pd.DataFrame(rows).to_csv(directory / "fold_metrics.csv", index=False)
        if not np.all(coverage == 1):
            raise ValueError("Every image must receive exactly one outer-test prediction")
        pd.DataFrame({"image_id": identifiers, "Class": y, "prediction": predictions,
                      "outer_fold": fold_numbers}).to_csv(directory / "oof_predictions.csv", index=False)
        write_json(directory / "classification_report.json", classification_report(
            y, predictions, labels=classes, output_dict=True, zero_division=0))
        _save_confusion(y, predictions, classes, directory)
        folds = pd.DataFrame(rows).drop(columns="fold")
        summaries[name] = {
            "outer_fold_summary": {metric: {"mean": folds[metric].mean(), "std_ddof0": folds[metric].std(ddof=0)} for metric in folds},
            "pooled_oof_metrics": _scores(y, predictions, classes),
        }
        write_json(directory / "summary.json", summaries[name])
        write_json(output_dir / "summary.json", summaries)

    if plan.settings["refit_full_data"]:
        for name in plan.settings["models"]:
            if progress:
                progress(f"final_refit/{name}")
            directory = output_dir / name / "final"
            directory.mkdir()
            fitted, selection = search(name, X, y, plan.final, directory)
            joblib.dump(fitted.best_estimator_, directory / "pipeline.joblib")
            write_json(directory / "metadata.json", {
                "purpose": "Full-data tuning and refit; not an independent evaluation",
                "feature_columns": features, "class_labels": classes,
                "estimator_classes": fitted.best_estimator_.classes_,
                "training_images": len(y), "library_versions": versions, **selection,
            })
    return summaries
