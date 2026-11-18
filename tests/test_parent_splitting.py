"""Parent mapping, grouped evaluation identity, and legacy training regression checks."""
import numpy as np
import pandas as pd
import pytest
from training.splitting import load_parent_groups, grouped_folds
from training.evaluation import prepare_evaluation, evaluate_nested_cv, save_split_ids
import json


def originals():
    # Each parent contributes both classes; groups must never be class-scoped.
    return pd.DataFrame([dict(image_id=f"{g}/{c}.png", parent_image_id=f"parent-{g}",
                              Class=c, value=float(g * 2 + i))
                         for g in range(20) for i, c in enumerate(["a", "b"])])


def settings():
    return dict(models=["svm", "knn"], outer_folds=3, inner_folds=3, n_jobs=1,
                param_grids={"svm": {"kernel": ["linear"]}, "knn": {"n_neighbors": [1]}},
                refit_full_data=True)


def plan(frame):
    return prepare_evaluation(settings(), frame.Class, image_ids=frame.image_id,
                              parent_ids=frame.parent_image_id)


def test_mapping_alignment_and_literal_na(tmp_path):
    frame = originals()
    frame.loc[0, "parent_image_id"] = "NA"
    path = tmp_path / "parents.csv"
    frame[["image_id", "parent_image_id"]].sample(frac=1, random_state=3).to_csv(path, index=False)
    pd.testing.assert_frame_equal(load_parent_groups(frame, path), frame)


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "blank_id", "blank_parent", "header"])
def test_bad_mapping(tmp_path, problem):
    frame = originals()
    mapping = frame[["image_id", "parent_image_id"]].copy()
    if problem == "missing": mapping = mapping.iloc[1:]
    elif problem == "extra": mapping.loc[len(mapping)] = ["unknown", "parent"]
    elif problem == "duplicate": mapping.loc[1, "image_id"] = mapping.loc[0, "image_id"]
    elif problem == "blank_id": mapping.loc[0, "image_id"] = " "
    elif problem == "blank_parent": mapping.loc[0, "parent_image_id"] = ""
    else: mapping = mapping.rename(columns={"parent_image_id": "parent"})
    path = tmp_path / "parents.csv"
    mapping.to_csv(path, index=False)
    with pytest.raises(ValueError): load_parent_groups(frame, path)


def test_nested_groups_and_artifacts(tmp_path):
    frame = originals()
    candidate = plan(frame)
    path = tmp_path / "splits.json"
    save_split_ids(candidate, frame.image_id, path)
    saved = path.read_bytes()
    save_split_ids(plan(frame), frame.image_id, path)
    assert saved == path.read_bytes()
    records = json.loads(saved)
    tests = []
    for outer in records["outer"]:
        tests.extend(outer["test_ids"])
        for record in [outer] + outer["inner"]:
            assert not set(record["train_parent_ids"]) & set(record["test_parent_ids"])
        for inner in outer["inner"]:
            assert set(inner["train_ids"] + inner["test_ids"]) == set(outer["train_ids"])
    assert sorted(tests) == sorted(frame.image_id)
    for record in records["full_data_tuning"]:
        assert not set(record["train_parent_ids"]) & set(record["test_parent_ids"])
    evaluate_nested_cv(frame, tmp_path / "evaluation", candidate)
    for model in ["svm", "knn"]:
        predictions = pd.read_csv(tmp_path / "evaluation" / model / "oof_predictions.csv")
        assert predictions.image_id.tolist() == frame.image_id.tolist()


@pytest.mark.parametrize("problem", ["reorder", "parent", "missing"])
def test_bound_identity_rejected_before_output(tmp_path, problem):
    frame = originals()
    candidate = plan(frame)
    if problem == "reorder": frame = frame.iloc[::-1]
    elif problem == "parent": frame.loc[0, "parent_image_id"] = "wrong"
    else: frame = frame.drop(columns="parent_image_id")
    with pytest.raises(ValueError): evaluate_nested_cv(frame, tmp_path / "bad", candidate)
    assert not (tmp_path / "bad").exists()


def test_group_shortage_and_knn_feasibility():
    frame = originals()
    with pytest.raises(ValueError, match="parent groups"):
        grouped_folds(frame.Class, ["one"] * len(frame), 3, 42, "Outer")
    options = settings()
    options["param_grids"]["knn"] = {"n_neighbors": [1000]}
    with pytest.raises(ValueError, match="smallest CV"):
        prepare_evaluation(options, frame.Class, image_ids=frame.image_id, parent_ids=frame.parent_image_id)


@pytest.mark.parametrize("name", ["svm", "knn"])
def test_legacy_grouped_search(name, monkeypatch):
    import matplotlib.pyplot as plt
    from training.classifiers import svm_model, knn_model
    from sklearn.model_selection import GridSearchCV
    module = svm_model if name == "svm" else knn_model
    estimator, _ = getattr(module, f"{name}_definition")()
    grid = {"kernel": ["linear"]} if name == "svm" else {"n_neighbors": [1]}
    monkeypatch.setattr(module, f"{name}_definition", lambda: (estimator, grid))
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    observed = []
    class Search(GridSearchCV):
        def fit(self, X, y, **kwargs):
            assert X.shape[1] == 1
            # Every retained parent contributes both original images.
            counts = np.unique(X[:, 0].astype(int) // 2, return_counts=True)[1]
            assert np.all(counts == 2)
            parents = np.array([f"parent-{int(v) // 2}" for v in X[:, 0]])
            for train, test in self.cv:
                assert not set(parents[train]) & set(parents[test])
            observed.append(len(self.cv))
            return super().fit(X, y, **kwargs)
    monkeypatch.setattr(module, "GridSearchCV", Search)
    frame = originals()
    getattr(module, f"run_{name}")(frame)
    assert observed == [10]
    with pytest.raises(ValueError, match="requires"):
        getattr(module, f"run_{name}")(frame.drop(columns="parent_image_id"))
