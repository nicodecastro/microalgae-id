"""Identity, strict alignment, classifier metadata, and audit-command checks."""

import csv
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from microalgae.imaging.dataset import (
    ExtractionFailureReport, build_dataset_manifest, failure_report_path, image_id,
)
from training.feature_tables import (
    audit_feature_tables, classifier_inputs, combine_feature_tables,
)


@pytest.fixture
def manifest():
    return pd.DataFrame({
        "image_id": ["b/shared.png", "a/shared.png", "a/last.png"],
        "Class": ["b", "a", "a"],
        "parent_image_id": ["", "", ""],
    })


def feature_table(manifest, values=(10., 20., 30.)):
    frame = manifest.drop(columns="parent_image_id").copy()
    frame.insert(0, "value", values)
    return frame


def test_manifest_order_extensions_root_move_and_distinct_same_names(tmp_path):
    root = tmp_path / "raw"
    for label in ["b", "a"]:
        (root / label).mkdir(parents=True)
        for name in ["z.png", "shared.JPG", "1.jpeg", "2.bmp", "3.tif", "4.TIFF", "ignored.txt"]:
            (root / label / name).touch()
        (root / label / "directory.png").mkdir()
    manifest = build_dataset_manifest(root, ["b", "a"])
    expected_names = ["1.jpeg", "2.bmp", "3.tif", "4.TIFF", "shared.JPG", "z.png"]
    assert manifest.image_id.tolist() == [f"{label}/{name}" for label in ["b", "a"] for name in expected_names]
    assert manifest.parent_image_id.eq("").all()
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    pd.testing.assert_frame_equal(manifest, build_dataset_manifest(moved, ["b", "a"]))
    assert image_id(root / "a" / "shared.JPG", root) != image_id(root / "b" / "shared.JPG", root)
    with pytest.raises(ValueError):
        image_id(tmp_path / "outside.png", root)


def test_join_uses_ids_not_row_position_and_preserves_values(manifest, tmp_path):
    first = feature_table(manifest)
    second = feature_table(manifest, (4., 5., 6.))
    tables = {"geom": first.iloc[[2, 0, 1]], "lbp": second.iloc[[1, 2, 0]]}
    assert audit_feature_tables(manifest, tables).empty
    output = tmp_path / "combined.csv"
    combined = combine_feature_tables(manifest, tables, output)
    assert combined.columns.tolist() == ["geom__value", "lbp__value", "image_id", "Class"]
    assert combined.image_id.tolist() == manifest.image_id.tolist()
    np.testing.assert_array_equal(combined["geom__value"], first.value)
    np.testing.assert_array_equal(combined["lbp__value"], second.value)
    pd.testing.assert_frame_equal(combined, pd.read_csv(output))
    pd.testing.assert_frame_equal(first, feature_table(manifest))


@pytest.mark.parametrize("problem,expected_issue", [
    ("legacy", "missing_column"), ("duplicate", "duplicate_image_id"),
    ("blank_id", "blank_image_id"), ("numeric_id", "invalid_image_id"),
    ("blank_label", "blank_Class"), ("conflict", "label_conflict"),
    ("missing", "missing_image"), ("extra", "unexpected_image"),
    ("nan", "nonfinite_features"), ("inf", "nonfinite_features"),
    ("text", "nonnumeric_features"), ("empty", "empty_table"),
    ("no_features", "no_features"), ("duplicate_columns", "duplicate_columns"),
])
def test_bad_tables_block_output_without_touching_previous_files(manifest, tmp_path, problem, expected_issue):
    frame = feature_table(manifest)
    if problem == "legacy":
        frame = frame.drop(columns="image_id")
    elif problem == "duplicate":
        frame.loc[1, "image_id"] = frame.loc[0, "image_id"]
    elif problem == "blank_id":
        frame.loc[0, "image_id"] = " "
    elif problem == "numeric_id":
        frame["image_id"] = frame["image_id"].astype(object)
        frame.loc[0, "image_id"] = 12
    elif problem == "blank_label":
        frame.loc[0, "Class"] = np.nan
    elif problem == "conflict":
        frame.loc[0, "Class"] = "wrong"
    elif problem == "missing":
        frame = frame.iloc[1:]
    elif problem == "extra":
        frame.loc[0, "image_id"] = "b/unknown.png"
    elif problem in ("nan", "inf"):
        frame.loc[0, "value"] = np.nan if problem == "nan" else np.inf
    elif problem == "text":
        frame["value"] = ["text", "2", "3"]
    elif problem == "empty":
        frame = frame.iloc[:0]
    elif problem == "no_features":
        frame = frame.drop(columns="value")
    elif problem == "duplicate_columns":
        frame = pd.concat([frame, frame[["value"]]], axis=1)
    issues = audit_feature_tables(manifest, {"geom": frame})
    assert expected_issue in set(issues.issue)
    output = tmp_path / "new" / "combined.csv"
    with pytest.raises(ValueError, match="Feature alignment failed"):
        combine_feature_tables(manifest, {"geom": frame}, output)
    assert not output.parent.exists()
    previous = tmp_path / "previous.csv"
    previous.write_text("existing data", encoding="utf-8")
    with pytest.raises(ValueError):
        combine_feature_tables(manifest, {"geom": frame}, previous)
    assert previous.read_text() == "existing data"


def test_audit_reports_multiple_independent_problems_including_legacy_labels(manifest, tmp_path):
    legacy = feature_table(manifest).drop(columns="image_id")
    legacy.loc[[0, 2], "Class"] = np.nan
    legacy.loc[1, "value"] = np.inf
    issues = audit_feature_tables(manifest, {"geom": legacy, "gabor": tmp_path / "missing.csv"})
    assert {"missing_column", "blank_Class", "nonfinite_features", "unreadable_table"} <= set(issues.issue)
    assert issues.loc[issues.issue == "blank_Class", "detail"].iloc[0].startswith("2 rows")


@pytest.mark.parametrize("source", [{}, [], None])
def test_empty_or_implicit_selection_rejected(manifest, source):
    assert "empty_selection" in audit_feature_tables(manifest, source).issue.tolist()


@pytest.mark.parametrize("problem", ["duplicate", "missing_id", "blank_label", "empty"])
def test_invalid_manifest_rejected(manifest, problem):
    frame = manifest.copy()
    if problem == "duplicate":
        frame.loc[1, "image_id"] = frame.loc[0, "image_id"]
    elif problem == "missing_id":
        frame = frame.drop(columns="image_id")
    elif problem == "blank_label":
        frame.loc[0, "Class"] = ""
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        combine_feature_tables(frame, {"geom": feature_table(manifest)})


def test_failed_retry_ledger_blocks_stale_table_then_resets(manifest, tmp_path):
    output = tmp_path / "feature.csv"
    feature_table(manifest).to_csv(output, index=False)
    original = output.read_bytes()
    failures = ExtractionFailureReport(output, tmp_path)
    with pytest.raises(OSError, match="preview failed"):
        with failures.track(tmp_path / "b" / "shared.png", "b"):
            raise OSError("preview failed")
    issues = audit_feature_tables(manifest, {"geom": output})
    assert issues.issue.tolist() == ["extraction_failure"]
    assert issues.image_id.tolist() == ["b/shared.png"]
    assert output.read_bytes() == original
    ExtractionFailureReport(output, tmp_path)
    assert audit_feature_tables(manifest, {"geom": output}).empty
    assert pd.read_csv(failure_report_path(output)).empty


def test_csv_duplicate_headers_and_malformed_failure_report(manifest, tmp_path):
    output = tmp_path / "feature.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(["value", "value", "image_id", "Class"])
    assert "unreadable_table" in audit_feature_tables(manifest, {"geom": output}).issue.tolist()
    feature_table(manifest).to_csv(output, index=False)
    pd.DataFrame({"unexpected": []}).to_csv(failure_report_path(output), index=False)
    assert "invalid_failure_report" in audit_feature_tables(manifest, {"geom": output}).issue.tolist()


def test_literal_na_identifier_and_label_survive_csv_read(tmp_path):
    manifest = pd.DataFrame({"image_id": ["NA"], "Class": ["NA"]})
    path = tmp_path / "table.csv"
    feature_table(manifest.assign(parent_image_id=""), (1.,)).to_csv(path, index=False)
    assert audit_feature_tables(manifest, {"geom": path}).empty


def test_classifier_accepts_legacy_and_keyed_inputs_without_metadata(manifest):
    frame = feature_table(manifest)
    frame["parent_image_id"] = ["parent", "", ""]
    X, y = classifier_inputs(frame)
    np.testing.assert_array_equal(X, [[10.], [20.], [30.]])
    np.testing.assert_array_equal(y, ["b", "a", "a"])
    old_X, old_y = classifier_inputs(frame.drop(columns=["image_id", "parent_image_id"]))
    np.testing.assert_array_equal(X, old_X)
    np.testing.assert_array_equal(y, old_y)


@pytest.mark.parametrize("problem", ["no_label", "blank_label", "nan", "text", "no_features"])
def test_classifier_rejects_invalid_data(manifest, problem):
    frame = feature_table(manifest)
    if problem == "no_label":
        frame = frame.drop(columns="Class")
    elif problem == "blank_label":
        frame.loc[0, "Class"] = " "
    elif problem == "nan":
        frame.loc[0, "value"] = np.nan
    elif problem == "text":
        frame["unexpected"] = "bad"
    else:
        frame = frame.drop(columns="value")
    with pytest.raises(ValueError):
        classifier_inputs(frame)


@pytest.mark.parametrize("classifier", ["svm", "knn"])
def test_real_classifier_entry_points_exclude_ids_before_split(manifest, monkeypatch, classifier):
    from training.classifiers import knn_model, svm_model

    module = svm_model if classifier == "svm" else knn_model

    def split(X, y, **kwargs):
        assert X.dtype.kind == "f"
        np.testing.assert_array_equal(X, [[10.], [20.], [30.]])
        assert kwargs == {"test_size": 0.2, "random_state": 42, "stratify": y}
        raise RuntimeError("stop before training")

    monkeypatch.setattr(module, "train_test_split", split)
    with pytest.raises(RuntimeError, match="stop before training"):
        getattr(module, f"run_{classifier}")(feature_table(manifest))


def test_audit_command_reports_legacy_then_combines_fresh_without_overwriting(tmp_path):
    from scripts.check_feature_tables import main

    root = tmp_path / "raw"
    (root / "a").mkdir(parents=True)
    (root / "a" / "image.png").touch()
    features = tmp_path / "old"
    features.mkdir()
    old_path = features / "Geometric_shape_features.csv"
    pd.DataFrame({"Area": [3.], "Class": [np.nan]}).to_csv(old_path, index=False)
    original = old_path.read_bytes()
    config = {
        "dataset": {"raw_dir": str(root), "classes": ["a"], "features_dir": str(features),
                    "combined_features_dir": str(tmp_path / "combined")},
        "features": {"feature_list": ["geom"]},
        "experiment": {"results_dir": str(tmp_path / "results")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["--config", str(config_path), "--combine"]) == 1
    output = tmp_path / "combined" / "Combined_features_keyed.csv"
    assert not output.exists()
    report_dir = tmp_path / "results" / "dataset_identity"
    issues = pd.read_csv(report_dir / "feature_table_audit.csv")
    assert {"missing_column", "blank_Class"} <= set(issues.issue)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    pd.DataFrame({"Area": [3.], "image_id": ["a/image.png"], "Class": ["a"]}).to_csv(
        fresh / old_path.name, index=False,
    )
    args = ["--config", str(config_path), "--features-dir", str(fresh)]
    assert main(args) == 0
    assert not output.exists()
    assert main([*args, "--combine"]) == 0
    assert pd.read_csv(output).columns.tolist() == ["geom__Area", "image_id", "Class"]
    assert old_path.read_bytes() == original
