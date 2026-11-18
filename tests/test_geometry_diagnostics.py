"""Reproducible, non-training diagnostic reports from a saved failed run."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from microalgae.imaging.dataset import build_dataset_manifest
from microalgae.imaging.preprocessing.configured import PreparedInput
from scripts import diagnose_geometry as diagnostic
from training.config import load_config
from training.extraction import extract_configured_features


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved_run(tmp_path):
    root = tmp_path / "raw"
    for label in ["b", "a"]:
        (root / label).mkdir(parents=True)
        for index in range(7):
            image = np.full((60, 90), 210, dtype=np.uint8)
            cv2.ellipse(image, (45, 30), (15 + index, 10), 25, 0, 360, 25, -1)
            assert cv2.imwrite(str(root / label / f"good_{index}.png"), image)
        assert cv2.imwrite(str(root / label / "failed.png"), np.full((60, 90), 180, dtype=np.uint8))
    config = load_config()
    config["dataset"].update(raw_dir=str(root), classes=["b", "a"])
    config["features"]["feature_list"] = ["geom"]
    run = tmp_path / "source_run"
    run.mkdir()
    build_dataset_manifest(root, ["b", "a"]).to_csv(run / "dataset_manifest.csv", index=False)
    with pytest.warns(UserWarning, match="Skipping"), pytest.raises(ValueError, match="audit failed"):
        extract_configured_features(config, run / "features", standardize=False)
    return run


def hashes(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_sampling_all_failures_and_five_controls_per_class_is_reproducible(saved_run):
    root, first, profiles = diagnostic._load_selection(saved_run)
    _, second, second_profiles = diagnostic._load_selection(saved_run)
    pd.testing.assert_frame_equal(first, second)
    assert profiles == second_profiles
    assert first.baseline_status.value_counts().to_dict() == {"control": 10, "failed": 2}
    assert first[first.baseline_status == "control"].groupby("Class").size().to_dict() == {"a": 5, "b": 5}
    manifest = pd.read_csv(saved_run / "dataset_manifest.csv")
    assert first.image_id.tolist() == manifest.loc[manifest.image_id.isin(first.image_id), "image_id"].tolist()
    assert list(profiles) == list(diagnostic.PROFILE_NAMES)
    assert not profiles["canny_baseline"]["foreground"]["enabled"]
    assert list(profiles) == ["canny_baseline", "otsu_mask", "otsu_bright_mask", "adaptive_mask",
                              "otsu_background_k31_mask", "otsu_background_k61_mask"]
    for kernel in (31, 61):
        assert profiles[f"otsu_background_k{kernel}_mask"]["foreground"]["background"]["kernel_size"] == kernel
    assert profiles["otsu_bright_mask"]["foreground"]["method"] == "otsu_bright"
    for name in ["otsu_mask", "otsu_bright_mask", "adaptive_mask"]:
        assert profiles[name]["foreground"]["enabled"]
        assert not profiles[name]["threshold"]["enabled"]
        assert not profiles[name]["segment"]["enabled"]
    assert root.is_dir()


def test_small_control_pools_use_all_available_images(saved_run):
    root, _, _ = diagnostic._load_selection(saved_run)
    # Produce a coherent smaller saved inventory rather than silently losing rows.
    manifest = pd.read_csv(saved_run / "dataset_manifest.csv")
    manifest = manifest[manifest.image_id.str.contains("failed|good_[01]", regex=True)]
    manifest.to_csv(saved_run / "dataset_manifest.csv", index=False)
    path = saved_run / "features" / "Geometric_shape_features.csv"
    frame = pd.read_csv(path)
    frame[frame.image_id.isin(manifest.image_id)].to_csv(path, index=False)
    _, selected, _ = diagnostic._load_selection(saved_run)
    assert len(selected) == 6
    assert selected.baseline_status.value_counts().to_dict() == {"control": 4, "failed": 2}


def test_hysteresis_profile_is_opt_in_and_preserves_all_existing_profiles(saved_run):
    root, selected, old = diagnostic._load_selection(saved_run)
    new_root, new_selected, profiles = diagnostic._load_selection(saved_run, include_hysteresis=True)
    assert root == new_root
    pd.testing.assert_frame_equal(selected, new_selected)
    assert {key: profiles[key] for key in old} == old
    assert len(profiles) == 7
    candidate = profiles["otsu_background_k61_hysteresis_mask"]["foreground"]
    assert candidate["method"] == "otsu_background_hysteresis"
    assert candidate["background"]["kernel_size"] == 61


def test_quality_profiles_are_opt_in_and_preserve_control(saved_run):
    _, selected, old = diagnostic._load_selection(saved_run)
    _, new_selected, profiles = diagnostic._load_selection(saved_run, include_quality=True)
    pd.testing.assert_frame_equal(selected, new_selected)
    assert {key: profiles[key] for key in old} == old
    assert len(profiles) == 9
    assert "otsu_background_k61_hysteresis_mask" not in profiles
    for name in diagnostic.QUALITY_PROFILE_NAMES:
        settings = profiles[name]
        assert settings["filter"]["enabled"] == ("bilateral" in name)
        if "bilateral" in name:
            assert settings["filter"]["bilateral"]["params"] == [5, 15, 3]
        assert settings["foreground"]["method"] == ("otsu_background_clahe" if "clahe" in name else "otsu_background")
        assert settings["foreground"]["background"]["kernel_size"] == 61


def test_selected_quality_settings_preserve_earlier_profiles(saved_run):
    _, selected, old = diagnostic._load_selection(saved_run, include_quality=True)
    _, actual, profiles = diagnostic._load_selection(saved_run, include_quality_tuning=True)
    pd.testing.assert_frame_equal(actual, selected)
    assert {name: profiles[name] for name in old} == old
    assert len(profiles) == 12
    assert "otsu_background_k61_hysteresis_mask" not in profiles
    b2 = profiles["otsu_background_k61_b2_mask"]
    assert b2["filter"]["bilateral"]["params"] == [5, 25, 3]
    assert b2["foreground"]["method"] == "otsu_background"
    for name, method in [("c1", "otsu_background_clahe_clip1"), ("c2", "otsu_background_clahe_grid2")]:
        settings = profiles[f"otsu_background_k61_{name}_mask"]
        assert not settings["filter"]["enabled"]
        assert settings["foreground"]["method"] == method
        assert settings["foreground"]["background"]["kernel_size"] == 61


def test_flags_are_warnings_and_degenerate_contours_remain_rejected():
    canvas = np.full((100, 100, 3), 190, dtype=np.uint8)
    content = np.zeros((100, 100), dtype=bool)
    content[10:90, 10:90] = True
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(image, (10, 45), 10, 255, -1)
    image[~content] = 0
    cv2.ellipse(image, (60, 60), (3, 2), 0, 0, 360, 255, -1)
    cv2.line(image, (30, 25), (40, 25), 255, 1)
    record, overlay = diagnostic._measure(PreparedInput(image), content, canvas)
    assert record["extraction_status"] == "measured"
    assert record["accepted_contour_count"] == 2
    assert record["contour_count"] == 3
    assert set(record["flags"].split(";")) == {"touches_content_border", "tiny_accepted_contour"}
    assert record["Area"] is not None
    assert overlay.shape == canvas.shape
    image[:] = 0
    record, _ = diagnostic._measure(PreparedInput(image), content, canvas)
    assert record["flags"] == "no_accepted_contours"
    assert all(record[name] is None for name in diagnostic.FEATURE_NAMES)


def test_full_command_saves_unapproved_reports_and_preserves_inputs(saved_run, tmp_path):
    root, selected, profiles = diagnostic._load_selection(saved_run)
    protected = list(saved_run.rglob("*.*")) + list(root.rglob("*.png"))
    before = hashes(protected)
    output = tmp_path / "comparison"
    command = [sys.executable, str(ROOT / "scripts" / "diagnose_geometry.py"),
               "--run-dir", str(saved_run), "--output-dir", str(output)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All reviews remain unreviewed" in result.stdout
    summary = read_json(output / "summary.json")
    assert summary["status"] == "completed" and summary["selected_images"] == 12
    assert summary["processing_errors"] == 0 and summary["review_rows"] == 72
    actual_selection = pd.read_csv(output / "selection.csv")
    pd.testing.assert_frame_equal(actual_selection, selected.reset_index(drop=True))
    measurements = pd.read_csv(output / "measurements.csv", keep_default_na=False)
    review = pd.read_csv(output / "review.csv", keep_default_na=False)
    assert len(measurements) == len(review) == 72
    references = pd.read_csv(output / "reference_selection.csv")
    assert references.image_id.tolist() == selected.image_id.tolist()
    assert not references.selected.any()
    assert len(read_json(output / "source_images.json")) == len(selected)
    assert set(review.review_status) == {"unreviewed"} and set(review.notes) == {""}
    assert pd.read_csv(output / "errors.csv").empty
    assert set(measurements.profile) == set(diagnostic.PROFILE_NAMES)
    failed = measurements[measurements.image_id.str.endswith("failed.png")]
    assert set(failed.extraction_status) == {"no_accepted_contours"}
    assert all((failed[name] == "").all() for name in diagnostic.FEATURE_NAMES)
    snapshots = yaml.safe_load((output / "resolved_settings.yaml").read_text())
    assert snapshots["profiles"] == profiles and snapshots["review_required"] is True
    for identifier in selected.image_id:
        path = output / "comparisons" / (identifier + "_comparison.png")
        assert cv2.imread(str(path)) is not None
        for profile in diagnostic.PROFILE_NAMES:
            preview = output / "previews" / profile / (identifier + "_image.png")
            assert cv2.imread(str(preview), 0) is not None
            if profile != "canny_baseline":
                mask = output / "previews" / profile / (identifier + "_mask.png")
                np.testing.assert_array_equal(cv2.imread(str(mask), 0), cv2.imread(str(preview), 0))
            if profile.startswith("otsu_background"):
                for stage in ("background", "residual"):
                    assert cv2.imread(str(output / "previews" / profile / (identifier + f"_{stage}.png")), 0) is not None
            assert (output / "overlays" / profile / (identifier + "_overlay.png")).exists()
    # Diagnostic tables cannot accidentally masquerade as standard feature exports.
    assert not (output / "Geometric_shape_features.csv").exists()
    assert not list(output.rglob("*.joblib"))
    assert hashes(protected) == before
    # Exercise the complete artifact handoff without requiring mouse input.
    from scripts._geometry_references import ReferenceSession
    from scripts.score_geometry_masks import score_geometry_masks

    references.selected = True
    references.to_csv(output / "reference_selection.csv", index=False)
    session = ReferenceSession(output, tmp_path / "references")
    session.save(0, [{"mode": "add", "vertices": [[90, 90], [165, 90], [165, 165], [90, 165]]}], "approved")
    session.save(1, [], "uncertain")
    scores = score_geometry_masks(output, session.output, tmp_path / "scores")
    assert len(scores) == 6 and scores.image_id.nunique() == 1
    assert hashes(protected) == before
    status_bytes = (output / "summary.json").read_bytes()
    rerun = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert rerun.returncode == 1 and "must be new" in rerun.stderr
    assert (output / "summary.json").read_bytes() == status_bytes


@pytest.mark.parametrize("problem", ["failure_id", "label_conflict", "duplicate", "missing_failure", "legacy", "not_canny", "escaping_id"])
def test_invalid_saved_inputs_write_nothing(saved_run, tmp_path, problem):
    ledger = saved_run / "features" / "Geometric_shape_features_failures.csv"
    failures = pd.read_csv(ledger)
    if problem == "failure_id":
        failures.loc[0, "image_id"] = "unknown/image.png"
    elif problem == "label_conflict":
        failures.loc[0, "Class"] = "wrong"
    elif problem == "duplicate":
        failures = pd.concat([failures, failures.iloc[:1]])
    elif problem == "missing_failure":
        failures = failures.iloc[1:]
    elif problem == "legacy":
        path = saved_run / "features" / "Geometric_shape_features.csv"
        pd.read_csv(path).drop(columns="image_id").to_csv(path, index=False)
    elif problem == "not_canny":
        path = saved_run / "features" / "extraction_settings.yaml"
        snapshot = yaml.safe_load(path.read_text())
        snapshot["resolved_preprocessing"]["geom"]["segment"]["enabled"] = False
        path.write_text(yaml.safe_dump(snapshot))
    else:
        identifier = failures.loc[0, "image_id"]
        path = saved_run / "dataset_manifest.csv"
        manifest = pd.read_csv(path)
        manifest.loc[manifest.image_id == identifier, "image_id"] = "../escaping.png"
        manifest.to_csv(path, index=False)
        failures.loc[0, "image_id"] = "../escaping.png"
    failures.to_csv(ledger, index=False)
    with pytest.raises(ValueError):
        diagnostic.diagnose_geometry(saved_run, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_raw_and_source_run_output_targets_rejected(saved_run):
    root, _, _ = diagnostic._load_selection(saved_run)
    for output in [root / "diagnostic", saved_run / "diagnostic"]:
        with pytest.raises(ValueError, match="outside"):
            diagnostic.diagnose_geometry(saved_run, output)
        assert not output.exists()


@pytest.mark.parametrize("problem", ["unreadable", "preview", "overlay", "comparison"])
def test_processing_errors_keep_reports_return_nonzero_and_never_fit_models(saved_run, tmp_path, monkeypatch, problem):
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    def forbidden(*args, **kwargs):
        pytest.fail("Diagnostics must never fit models or standardize datasets")

    monkeypatch.setattr(GridSearchCV, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    original_load = diagnostic._load_selection
    root, selected, profiles = original_load(saved_run)
    selected = selected.head(1)
    monkeypatch.setattr(diagnostic, "_load_selection", lambda _, **kwargs: (root, selected, profiles))
    monkeypatch.setattr(diagnostic, "_save_comparison", lambda *args: None)
    if problem == "unreadable":
        (root / selected.image_id.iloc[0]).write_bytes(b"broken image")
    else:
        def failed_write(*args, **kwargs):
            raise OSError("Diagnostic write failed deliberately")
        monkeypatch.setattr(diagnostic, {
            "preview": "save_prepared_input", "overlay": "_save_png", "comparison": "_save_comparison",
        }[problem], failed_write)
    output = tmp_path / "errors"
    assert diagnostic.main(["--run-dir", str(saved_run), "--output-dir", str(output)]) == 1
    errors = pd.read_csv(output / "errors.csv")
    assert not errors.empty and errors.reason.notna().all()
    assert read_json(output / "summary.json")["status"] == "completed_with_errors"
    review = pd.read_csv(output / "review.csv")
    assert set(review.review_status) == {"unreviewed"}


def test_real_png_write_failure_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic.cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError, match="Failed to save"):
        diagnostic._save_png(tmp_path / "preview.png", np.zeros((5, 5), dtype=np.uint8))
