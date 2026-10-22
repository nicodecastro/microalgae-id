"""Nested-CV isolation, reproducibility, artifacts, and command integration."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
import yaml

from training.config import load_config
from training.evaluation import evaluate_nested_cv, prepare_evaluation, save_split_ids
from training.experiments import runner
from training.extraction import extract_configured_features
from training.feature_tables import FEATURE_FILENAMES, classifier_inputs


ROOT = Path(__file__).resolve().parents[1]


def tiny_settings():
    return {
        "models": ["svm", "knn"], "outer_folds": 5, "inner_folds": 5,
        "random_state": 42, "scoring": "f1_macro", "n_jobs": 1,
        "refit_full_data": True,
        "param_grids": {
            "svm": {"C": [1], "kernel": ["linear"]},
            "knn": {"n_neighbors": [1], "algorithm": ["ball_tree"]},
        },
    }


@pytest.fixture
def frame():
    rng = np.random.default_rng(18)
    return pd.DataFrame({
        "family__marker": np.arange(30, dtype=float),
        "family__value": rng.normal(size=30),
        "image_id": [f"{label}/{i}.png" for label in ["c", "a", "b"] for i in range(10)],
        "Class": np.repeat(["c", "a", "b"], 10),
        "parent_image_id": "",  # Must never enter a scaler or classifier.
    })


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_default_grids_and_feasibility_are_not_silently_reduced():
    labels = np.repeat(["a", "b", "c"], 100)
    plan = prepare_evaluation({}, labels)
    assert plan.settings["outer_folds"] == plan.settings["inner_folds"] == 5
    assert plan.settings["scoring"] == "f1_macro"
    assert plan.settings["random_state"] == 42
    assert plan.settings["n_jobs"] == -1
    svm_grid = plan.specifications["svm"][1]
    knn_grid = plan.specifications["knn"][1]
    assert len(ParameterGrid(svm_grid)) == 320
    assert len(ParameterGrid(knn_grid)) == 336
    assert svm_grid == {"C": [10**i for i in range(-5, 5)], "kernel": ["linear", "poly", "rbf", "sigmoid"],
                        "degree": [2, 3, 4, 5], "gamma": ["auto", "scale"]}
    assert knn_grid == {"n_neighbors": list(range(1, 42, 2)), "weights": ["uniform", "distance"],
                        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"], "p": [1, 2]}
    with pytest.raises(ValueError, match="smallest CV training set"):
        prepare_evaluation({}, np.repeat(["a", "b", "c"], 10))
    assert len(plan.specifications["knn"][1]["n_neighbors"]) == 21


@pytest.mark.parametrize("change, message", [
    ({"outer_folds": 1}, "at least 2"), ({"inner_folds": True}, "integer"),
    ({"n_jobs": 0}, "zero"), ({"random_state": -1}, "random_state"),
    ({"models": ["svm", "svm"]}, "unique"), ({"models": ["bad"]}, "unique"),
    ({"scoring": "not_a_scorer"}, "scoring"), ({"refit_full_data": 1}, "Boolean"),
    ({"cross_validation": "5-fold"}, "Unknown"),
    ({"param_grids": {"other": {"C": [1]}}}, "selected model"),
    ({"param_grids": {"svm": {"bad": [1]}}}, "Invalid parameter"),
    ({"param_grids": {"svm": {"C": [-1]}}}, "parameter"),
    ({"param_grids": {"svm": {"C": []}}}, "non-empty"),
])
def test_invalid_settings_fail_before_fitting(frame, change, message):
    with pytest.raises(ValueError, match=message):
        prepare_evaluation(tiny_settings() | change, frame.Class.to_numpy())


def test_insufficient_outer_and_inner_class_counts():
    with pytest.raises(ValueError, match="Outer CV needs"):
        prepare_evaluation(tiny_settings(), np.repeat(["a", "b"], 4))
    with pytest.raises(ValueError, match="inner CV needs"):
        prepare_evaluation(tiny_settings(), np.repeat(["a", "b"], 5))
    with pytest.raises(ValueError, match="two classes"):
        prepare_evaluation(tiny_settings(), np.repeat("a", 20))


def test_shared_splits_are_reproducible_and_isolated(frame, tmp_path):
    plan = prepare_evaluation(tiny_settings(), frame.Class.to_numpy())
    other = prepare_evaluation(tiny_settings(), frame.Class.to_numpy())
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    for candidate, path in zip([plan, other], paths):
        save_split_ids(candidate, frame.image_id.to_numpy(), path)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    splits = read_json(paths[0])
    assert splits["shared_by_models"] == ["svm", "knn"]
    all_test = []
    for fold in splits["outer"]:
        train, test = set(fold["train_ids"]), set(fold["test_ids"])
        assert not train & test
        all_test.extend(test)
        inner_tests = []
        for inner in fold["inner"]:
            assert set(inner["train_ids"]) | set(inner["test_ids"]) == train
            assert not set(inner["train_ids"]) & set(inner["test_ids"])
            assert not (set(inner["train_ids"]) | set(inner["test_ids"])) & test
            inner_tests.extend(inner["test_ids"])
        assert sorted(inner_tests) == sorted(train)
    assert sorted(all_test) == sorted(frame.image_id)


def test_scaler_fit_boundaries_oof_metrics_and_final_reload(frame, tmp_path, monkeypatch):
    plan = prepare_evaluation(tiny_settings(), frame.Class.to_numpy())
    observed = []
    original_fit = StandardScaler.fit

    def spy(self, values, *args, **kwargs):
        assert values.shape[1] == 2
        observed.append(tuple(values[:, 0].astype(int)))
        return original_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", spy)
    stages = []
    result = evaluate_nested_cv(frame, tmp_path / "evaluation", plan, progress=stages.append)
    expected = []
    for name in ["svm", "knn"]:
        for (train, _), inner in zip(plan.outer, plan.inner):
            expected.extend(tuple(train[inner_train]) for inner_train, _ in inner)
            expected.append(tuple(train))  # Refit on outer training only.
    for name in ["svm", "knn"]:
        expected.extend(tuple(train) for train, _ in plan.final)
        expected.append(tuple(range(len(frame))))  # Only final pipelines see all rows.
    assert observed == expected
    assert len(stages) == 12
    assert all(stage.startswith("nested_cv/") for stage in stages[:10])
    assert stages[10:] == ["final_refit/svm", "final_refit/knn"]
    X, y = classifier_inputs(frame)
    for name in ["svm", "knn"]:
        directory = tmp_path / "evaluation" / name
        oof = pd.read_csv(directory / "oof_predictions.csv")
        assert oof.image_id.tolist() == frame.image_id.tolist()
        assert oof.Class.tolist() == y.tolist()
        assert not oof.image_id.duplicated().any()
        assert oof.outer_fold.value_counts().tolist() == [6] * 5
        folds = pd.read_csv(directory / "fold_metrics.csv")
        summary = read_json(directory / "summary.json")
        assert summary == result[name]
        for metric in folds.drop(columns="fold"):
            assert summary["outer_fold_summary"][metric]["mean"] == pytest.approx(folds[metric].mean())
            assert summary["outer_fold_summary"][metric]["std_ddof0"] == pytest.approx(folds[metric].std(ddof=0))
        assert summary["pooled_oof_metrics"]["accuracy"] == accuracy_score(y, oof.prediction)
        assert summary["pooled_oof_metrics"]["macro_f1"] == pytest.approx(f1_score(y, oof.prediction, average="macro"))
        assert (directory / "confusion_matrix.png").read_bytes().startswith(b"\x89PNG")
        assert pd.read_csv(directory / "confusion_matrix.csv", index_col=0).to_numpy().sum() == len(frame)
        assert read_json(directory / "classification_report.json")["macro avg"]["support"] == len(frame)
        for fold in range(1, 6):
            evaluation = read_json(directory / f"outer_{fold}" / "evaluation.json")
            assert "inner_selection_score" in evaluation
            assert "outer_test_metrics" in evaluation
        model_path = directory / "final" / "pipeline.joblib"
        pipeline = joblib.load(model_path)
        np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, X.mean(axis=0))
        assert pipeline.named_steps["scaler"].n_samples_seen_ == len(frame)
        np.testing.assert_array_equal(pipeline.predict(X), joblib.load(model_path).predict(X))
        metadata = read_json(directory / "final" / "metadata.json")
        assert metadata["feature_columns"] == ["family__marker", "family__value"]
        assert metadata["class_labels"] == ["c", "a", "b"]
        assert metadata["training_images"] == len(frame)
        assert "not an independent evaluation" in metadata["purpose"]
        assert metadata["library_versions"]["scikit-learn"]
        assert len(pd.read_csv(directory / "final" / "search_results.csv")) == 1


def test_repeated_evaluations_reproduce_predictions_and_parameters(frame, tmp_path):
    plan = prepare_evaluation(tiny_settings(), frame.Class.to_numpy())
    for folder in ["one", "two"]:
        evaluate_nested_cv(frame, tmp_path / folder, plan)
    for name in ["svm", "knn"]:
        for filename in ["oof_predictions.csv", "summary.json", "final/selection.json"]:
            assert (tmp_path / "one" / name / filename).read_bytes() == (tmp_path / "two" / name / filename).read_bytes()


def test_parallel_inner_search_and_multiple_candidates(frame, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from training.classifiers import knn_model, svm_model

    def forbidden(*args, **kwargs):
        pytest.fail("Nested evaluation must not invoke interactive or holdout workflows")

    monkeypatch.setattr(plt, "show", forbidden)
    monkeypatch.setattr(knn_model, "run_knn", forbidden)
    monkeypatch.setattr(svm_model, "run_svm", forbidden)
    settings = tiny_settings() | {"models": ["svm"], "n_jobs": 2,
                                  "param_grids": {"svm": {"C": [0.1, 1], "kernel": ["linear"]}}}
    plan = prepare_evaluation(settings, frame.Class.to_numpy())
    evaluate_nested_cv(frame, tmp_path / "parallel", plan)
    results = pd.read_csv(tmp_path / "parallel" / "svm" / "final" / "search_results.csv")
    assert len(results) == 2
    assert set(results.param_model__C) == {0.1, 1}
    settings["n_jobs"] = 1
    evaluate_nested_cv(frame, tmp_path / "serial", prepare_evaluation(settings, frame.Class.to_numpy()))
    for filename in ["oof_predictions.csv", "final/selection.json"]:
        assert (tmp_path / "parallel" / "svm" / filename).read_bytes() == (tmp_path / "serial" / "svm" / filename).read_bytes()


@pytest.mark.parametrize("problem", ["missing_id", "duplicate_id", "blank_id", "nonfinite", "class_order"])
def test_evaluation_rejects_bad_inputs_before_writing(frame, tmp_path, problem):
    plan = prepare_evaluation(tiny_settings(), frame.Class.to_numpy())
    if problem == "missing_id":
        frame = frame.drop(columns="image_id")
    elif problem == "duplicate_id":
        frame.loc[1, "image_id"] = frame.loc[0, "image_id"]
    elif problem == "blank_id":
        frame.loc[0, "image_id"] = " "
    elif problem == "nonfinite":
        frame.loc[0, "family__marker"] = np.nan
    else:
        frame = frame.iloc[::-1]
    with pytest.raises(ValueError):
        evaluate_nested_cv(frame, tmp_path / "invalid", plan)
    assert not (tmp_path / "invalid").exists()


def test_explicitly_disabled_final_refit(frame, tmp_path):
    settings = tiny_settings() | {"models": ["svm"], "param_grids": {"svm": {"C": [1]}}, "refit_full_data": False}
    plan = prepare_evaluation(settings, frame.Class.to_numpy())
    evaluate_nested_cv(frame, tmp_path / "evaluation", plan)
    assert not (tmp_path / "evaluation" / "svm" / "final").exists()
    assert read_json(tmp_path / "evaluation" / "splits.json")["full_data_tuning"] == []


@pytest.fixture
def experiment_config(tmp_path):
    config = load_config()
    raw = tmp_path / "raw"
    y, x = np.indices((32, 40))
    for label in ["b", "a"]:
        (raw / label).mkdir(parents=True)
        for i in range(10):
            image = ((x * (i + 1) + y * (3 if label == "a" else 7)) % 256).astype(np.uint8)
            assert cv2.imwrite(str(raw / label / f"{i}.png"), image)
    config["dataset"].update(raw_dir=str(raw), classes=["b", "a"],
                             features_dir=str(tmp_path / "legacy"), combined_features_dir=str(tmp_path / "old_combined"))
    config["features"]["feature_list"] = ["lbp", "glcm"]
    config["experiment"] = tiny_settings()
    config["logging"]["save_intermediate_images"] = True
    return config


def test_full_command_extracts_raw_keyed_features_and_saves_models(experiment_config, tmp_path):
    config = experiment_config
    original = deepcopy(config)
    raw_files = list(Path(config["dataset"]["raw_dir"]).rglob("*.png"))
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw_files}
    legacy = Path(config["dataset"]["features_dir"])
    legacy.mkdir()
    existing = legacy / "keep.csv"
    existing.write_text("do,not,change\n1,2,3\n")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "experiment"
    command = [sys.executable, str(ROOT / "main.py"), "--config", str(path), "--output-dir", str(output)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Experiment completed" in result.stdout
    assert read_json(output / "run_status.json")["status"] == "completed"
    combined = pd.read_csv(output / "Combined_features_keyed.csv")
    assert combined.image_id.tolist() == [f"{label}/{i}.png" for label in ["b", "a"] for i in range(10)]
    assert combined.columns[-2:].tolist() == ["image_id", "Class"]
    assert pd.read_csv(output / "feature_table_audit.csv").empty
    snapshot = yaml.safe_load((output / "resolved_settings.yaml").read_text())
    assert snapshot["standardize"] is False
    assert snapshot["configuration"] == config
    assert snapshot["resolved_experiment"]["param_grids"] == config["experiment"]["param_grids"]
    assert yaml.safe_load((output / "features" / "extraction_settings.yaml").read_text())["standardize"] is False
    # Compare raw CSVs to an independent extraction-only call, not scaled data.
    expected = extract_configured_features(config, tmp_path / "raw_expected", standardize=False)
    for family, dataframe in expected.items():
        actual = pd.read_csv(output / "features" / FEATURE_FILENAMES[family])
        pd.testing.assert_frame_equal(actual, dataframe)
        assert (output / "features" / "previews" / family / "a" / "0.png_image.png").exists()
    for model in ["svm", "knn"]:
        assert (output / "evaluation" / model / "final" / "pipeline.joblib").exists()
    assert config == original
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw_files} == hashes
    assert existing.read_text() == "do,not,change\n1,2,3\n"
    assert not Path(config["dataset"]["combined_features_dir"]).exists()
    # Re-running must fail and preserve the entire old run.
    saved_status = (output / "run_status.json").read_bytes()
    rerun = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert rerun.returncode == 1
    assert "must be new" in rerun.stderr
    assert (output / "run_status.json").read_bytes() == saved_status


@pytest.mark.parametrize("problem", ["extraction", "alignment", "training"])
def test_failures_keep_status_and_never_train_after_bad_alignment(experiment_config, tmp_path, monkeypatch, problem):
    config = experiment_config
    config["logging"]["save_intermediate_images"] = False
    output = tmp_path / "failed"
    calls = []

    def fail_training(*args, **kwargs):
        calls.append("training")
        raise RuntimeError("training failed deliberately")

    monkeypatch.setattr(runner, "evaluate_nested_cv", fail_training)
    if problem == "extraction":
        (Path(config["dataset"]["raw_dir"]) / "a" / "broken.png").touch()
    elif problem == "alignment":
        original_extract = runner.extract_configured_features

        def corrupt(*args, **kwargs):
            frames = original_extract(*args, **kwargs)
            path = Path(args[1]) / FEATURE_FILENAMES["lbp"]
            pd.read_csv(path).drop(index=0).to_csv(path, index=False)
            return frames

        monkeypatch.setattr(runner, "extract_configured_features", corrupt)
    with pytest.raises((ValueError, RuntimeError)):
        runner.run_configured_experiment(config, output)
    status = read_json(output / "run_status.json")
    assert status["status"] == "failed"
    assert status["error_type"] and status["reason"] and status["traceback"]
    assert calls == (["training"] if problem == "training" else [])
    if problem != "training":
        assert not (output / "Combined_features_keyed.csv").exists()
    if problem == "extraction":
        assert (output / "features" / "LBP_features_failures.csv").exists()
        assert (output / "features" / "feature_table_audit.csv").exists()
    if problem == "alignment":
        assert "missing_image" in pd.read_csv(output / "feature_table_audit.csv").issue.tolist()


def test_raw_target_and_preflight_failures_write_nothing(experiment_config, tmp_path):
    raw_output = Path(experiment_config["dataset"]["raw_dir"]) / "run"
    with pytest.raises(ValueError, match="raw dataset"):
        runner.run_configured_experiment(experiment_config, raw_output)
    assert not raw_output.exists()
    experiment_config["experiment"]["inner_folds"] = 20
    with pytest.raises(ValueError, match="inner CV"):
        runner.run_configured_experiment(experiment_config, tmp_path / "infeasible")
    assert not (tmp_path / "infeasible").exists()


def test_main_requires_output_directory_and_reports_errors(experiment_config, tmp_path, capsys):
    from main import main

    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(experiment_config), encoding="utf-8")
    assert main(["--config", str(path), "--output-dir", str(tmp_path)]) == 1
    assert "must be new" in capsys.readouterr().err
