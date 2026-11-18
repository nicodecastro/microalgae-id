"""Reference annotation persistence and paired mask scores without opening a GUI."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import yaml

from scripts._geometry_references import ReferenceSession, digest, mask_path, rasterize, read_binary
from scripts.annotate_geometry import AnnotationWindow
from scripts.diagnose_geometry import PROFILE_NAMES, QUALITY_PROFILE_NAMES, QUALITY_TUNING_PROFILES, _reference_canvas
from scripts.score_geometry_masks import error_overlay, overlap_scores, score_geometry_masks
from microalgae.imaging.preprocessing.configured import validate_preprocessing


ROOT = Path(__file__).resolve().parents[1]
POLYGONS = [{"mode": "add", "vertices": [[15, 22], [35, 22], [35, 38], [15, 38]]}]


@pytest.fixture
def comparison(tmp_path):
    raw = tmp_path / "raw"
    (raw / "a").mkdir(parents=True)
    directory = tmp_path / "comparison"
    directory.mkdir()
    baseline = validate_preprocessing({"resize": {"target_size": [64, 64]}}, "geom")
    sources, rows, measurements = [], [], []
    for index in range(12):
        identifier = f"a/{index}.png"
        image = np.full((30, 50, 3), 190, np.uint8)
        cv2.ellipse(image, (25, 15), (12, 6), 0, 0, 360, (50, 50, 50), -1)
        assert cv2.imwrite(str(raw / identifier), image)
        canvas, content = _reference_canvas(image, baseline)
        sources.append({"image_id": identifier, "sha256": digest(raw / identifier), "canvas_shape": list(content.shape)})
        rows.append({"image_id": identifier, "Class": "a", "baseline_status": "failed"})
        mask = rasterize(POLYGONS, content.shape).astype(np.uint8) * 255
        for profile in PROFILE_NAMES:
            folder = directory / "previews" / profile / "a"
            folder.mkdir(parents=True, exist_ok=True)
            predicted = cv2.Canny(mask, 50, 100) if profile == "canny_baseline" else mask
            assert cv2.imwrite(str(folder / f"{index}.png_image.png"), predicted)
            measurements.append({"image_id": identifier, "profile": profile, "extraction_status": "measured"})
    settings = {"dataset_dir": str(raw), "source_run": str(tmp_path / "source_run"),
                "profiles": {profile: baseline for profile in PROFILE_NAMES}}
    (directory / "resolved_settings.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    (directory / "source_images.json").write_text(json.dumps(sources), encoding="utf-8")
    pd.DataFrame(rows).to_csv(directory / "selection.csv", index=False)
    pd.DataFrame(rows).assign(selected=True).to_csv(directory / "reference_selection.csv", index=False)
    pd.DataFrame(measurements).to_csv(directory / "measurements.csv", index=False)
    return directory


def test_polygon_operations_multiple_objects_subtraction_and_validation():
    polygons = deepcopy(POLYGONS)
    polygons.append({"mode": "subtract", "vertices": [[20, 25], [30, 25], [30, 30], [20, 30]]})
    polygons.append({"mode": "add", "vertices": [[45, 25], [50, 25], [50, 30], [45, 30]]})
    mask = rasterize(polygons, (64, 64))
    assert mask[23, 16] and not mask[27, 25] and mask[27, 47]
    assert not rasterize([], (64, 64)).any()
    for vertices in ([[1, 2]], [[-1, 0], [2, 0], [2, 2]], [[1, 2], [2, 3], [float("nan"), 2]]):
        with pytest.raises(ValueError, match="polygon"):
            rasterize([{"mode": "add", "vertices": vertices}], (64, 64))


def test_session_save_uncertain_resume_and_explicit_overwrite_protection(comparison, tmp_path):
    output = tmp_path / "references"
    session = ReferenceSession(comparison, output)
    session.save(0, POLYGONS, "approved")
    session.save(1, [], "uncertain")
    before = {path: digest(path) for path in output.rglob("*.*")}
    resumed = ReferenceSession(comparison, output, resume=True)
    assert resumed.data == session.data
    assert resumed.data["images"][1]["status"] == "uncertain"
    assert {path: digest(path) for path in before} == before
    with pytest.raises(FileExistsError, match="resume"):
        ReferenceSession(comparison, output)
    with pytest.raises(ValueError, match="nonempty"):
        resumed.save(2, [], "approved")
    with pytest.raises(ValueError, match="padding"):
        resumed.save(2, [{"mode": "add", "vertices": [[0, 0], [10, 0], [10, 10], [0, 10]]}], "approved")
    assert resumed.data["images"][2]["status"] == "pending"
    np.testing.assert_array_equal(read_binary(mask_path(output, "a/0.png"), (64, 64)), rasterize(POLYGONS, (64, 64)))


@pytest.mark.parametrize("problem", ["count", "duplicate", "invalid_flag", "unknown_id", "missing_id"])
def test_reference_selection_invalid_writes_nothing(comparison, tmp_path, problem):
    path = comparison / "reference_selection.csv"
    frame = pd.read_csv(path)
    if problem == "count":
        frame["selected"] = False
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "invalid_flag":
        frame["selected"] = "maybe"
    elif problem == "unknown_id":
        frame.loc[0, "image_id"] = "../outside.png"
    else:
        frame = frame.iloc[1:]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        ReferenceSession(comparison, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("problem", ["source_hash", "settings", "identity", "mask", "status"])
def test_resume_and_scoring_reject_changed_provenance(comparison, tmp_path, problem):
    session = ReferenceSession(comparison, tmp_path / "references")
    session.save(0, POLYGONS, "approved")
    if problem == "source_hash":
        (tmp_path / "raw" / "a" / "0.png").write_bytes(b"changed")
    elif problem == "settings":
        with (comparison / "resolved_settings.yaml").open("a", encoding="utf-8") as stream:
            stream.write("\n# changed\n")
    elif problem == "mask":
        assert cv2.imwrite(str(mask_path(session.output, "a/0.png")), np.zeros((64, 64), np.uint8))
    else:
        data = deepcopy(session.data)
        data["images"][0]["image_id" if problem == "identity" else "status"] = "other"
        session.path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        score_geometry_masks(comparison, session.output, tmp_path / "scores")
    assert not (tmp_path / "scores").exists()


def test_overlap_known_identical_disjoint_and_partial_masks():
    reference = np.array([[True, True], [False, False]])
    content = np.ones_like(reference)
    identical = overlap_scores(reference, reference, content)
    assert identical == {"iou": 1.0, "dice": 1.0, "precision": 1.0, "recall": 1.0,
                         "true_positive_pixels": 2, "false_negative_pixels": 0, "false_positive_pixels": 0}
    disjoint = overlap_scores(~reference, reference, content)
    assert disjoint["precision"] == disjoint["recall"] == disjoint["iou"] == disjoint["dice"] == 0
    actual = overlap_scores(np.array([[True, False], [True, False]]), reference, content)
    assert actual["iou"] == pytest.approx(1 / 3) and actual["dice"] == 0.5
    assert actual["precision"] == actual["recall"] == 0.5
    empty = overlap_scores(np.zeros_like(reference), reference, content)
    assert empty["precision"] == empty["recall"] == empty["iou"] == empty["dice"] == 0
    asymmetric = overlap_scores(np.ones_like(reference), reference, content)
    assert asymmetric["precision"] == 0.5 and asymmetric["recall"] == 1
    with pytest.raises(ValueError, match="aligned"):
        overlap_scores(reference[:1], reference, content)
    content[1] = False
    assert overlap_scores(np.ones_like(reference), reference, content)["iou"] == 1
    assert overlap_scores(np.ones_like(reference), reference, content)["false_positive_pixels"] == 0


def test_error_overlay_colors_exclude_padding_and_preserve_inputs():
    canvas = np.full((2, 3, 3), 100, np.uint8)
    reference = np.array([[True, True, False], [False, False, False]])
    predicted = np.array([[True, False, True], [False, False, True]])
    content = np.array([[True, True, True], [True, True, False]])
    overlay = error_overlay(canvas, predicted, reference, content)
    np.testing.assert_array_equal(overlay[0], [[35, 165, 35], [178, 35, 178], [35, 126, 201]])
    np.testing.assert_array_equal(overlay[1], canvas[1])
    assert (canvas == 100).all()


def test_scoring_command_paired_metrics_exclusions_and_input_preservation(comparison, tmp_path):
    session = ReferenceSession(comparison, tmp_path / "references")
    session.save(0, POLYGONS, "approved")
    session.save(1, [], "uncertain")
    protected = list(comparison.rglob("*.*")) + list(session.output.rglob("*.*")) + list((tmp_path / "raw").rglob("*.png"))
    before = {path: digest(path) for path in protected}
    output = tmp_path / "scores"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/score_geometry_masks.py"),
                             "--comparison-dir", str(comparison), "--reference-dir", str(session.output),
                             "--output-dir", str(output)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(output / "per_image_scores.csv")
    assert len(frame) == 6 and frame.image_id.unique().tolist() == ["a/0.png"]
    assert (frame.loc[frame.profile != "canny_baseline", "iou"] == 1).all()
    assert set(frame.loc[frame.profile == "canny_baseline", "measurement_type"]) == {"edge_input_overlap"}
    assert set(frame.loc[frame.profile != "canny_baseline", "iou_difference_vs_dark_otsu"]) == {0}
    summary = pd.read_csv(output / "score_summary.csv")
    assert (summary.scored_count == 1).all() and (summary.excluded_count == 11).all()
    assert len(pd.read_csv(output / "excluded_references.csv")) == 11
    assert (frame.loc[frame.profile != "canny_baseline", ["precision", "recall"]] == 1).all().all()
    assert len(list((output / "error_overlays").rglob("*.png"))) == 6
    gallery = cv2.imread(str(output / "error_comparisons/a/0.png_comparison.png"))
    assert gallery is not None and gallery.shape[:2] == (1300, 1300)
    assert pd.read_csv(output / "review_order.csv").image_id.tolist() == ["a/0.png"]
    assert {path: digest(path) for path in protected} == before
    with pytest.raises(FileExistsError):
        score_geometry_masks(comparison, session.output, output)


@pytest.mark.parametrize("problem", ["missing", "shape", "nonbinary", "measurement", "duplicate"])
def test_missing_or_mismatched_predictions_fail_without_scores(comparison, tmp_path, problem):
    session = ReferenceSession(comparison, tmp_path / "references")
    session.save(0, POLYGONS, "approved")
    path = comparison / "previews/otsu_mask/a/0.png_image.png"
    if problem == "missing":
        path.unlink()
    elif problem in ("shape", "nonbinary"):
        image = np.zeros((3, 3), np.uint8) if problem == "shape" else np.full((64, 64), 4, np.uint8)
        assert cv2.imwrite(str(path), image)
    else:
        path = comparison / "measurements.csv"
        frame = pd.read_csv(path)
        frame = frame.iloc[1:] if problem == "measurement" else pd.concat([frame, frame.iloc[:1]])
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        score_geometry_masks(comparison, session.output, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_headless_editor_buttons_polygon_callback_undo_reset_and_saves(comparison, tmp_path):
    import matplotlib.pyplot as plt

    session = ReferenceSession(comparison, tmp_path / "references")
    window = AnnotationWindow(session)
    try:
        assert len(window.buttons) == 8
        window.start_polygon("add")
        window.selector.onselect(POLYGONS[0]["vertices"])
        assert window.polygons == POLYGONS
        window.undo()
        assert not window.polygons
        window.start_polygon("add")
        window.selector.onselect(POLYGONS[0]["vertices"])
        window.save("approved")
        assert window.index == 1 and session.data["images"][0]["status"] == "approved"
        window.save("uncertain")
        assert session.data["images"][1]["status"] == "uncertain"
        window.previous()
        window.previous()
        assert window.polygons == POLYGONS
        window.reset()
        window.save("approved")
        assert "nonempty" in window.message.get_text()
        assert session.data["images"][0]["polygons"] == POLYGONS
    finally:
        plt.close(window.figure)


def test_no_approved_masks_and_unsafe_output_targets_fail(comparison, tmp_path):
    session = ReferenceSession(comparison, tmp_path / "references")
    with pytest.raises(ValueError, match="No approved"):
        score_geometry_masks(comparison, session.output, tmp_path / "scores")
    assert not (tmp_path / "scores").exists()
    with pytest.raises(ValueError, match="outside"):
        ReferenceSession(comparison, tmp_path / "raw" / "new")
    session.save(0, POLYGONS, "approved")
    with pytest.raises(ValueError, match="outside"):
        score_geometry_masks(comparison, session.output, comparison / "new")


def test_mask_save_failure_retains_previous_decision(comparison, tmp_path, monkeypatch):
    import scripts._geometry_references as references

    session = ReferenceSession(comparison, tmp_path / "references")
    before = session.path.read_bytes()
    monkeypatch.setattr(references, "_save_png", lambda *args: (_ for _ in ()).throw(OSError("write failed")))
    with pytest.raises(OSError):
        session.save(0, POLYGONS, "approved")
    assert session.path.read_bytes() == before
    assert session.data["images"][0]["status"] == "pending"


def test_metadata_save_failure_restores_previous_mask_and_decision(comparison, tmp_path, monkeypatch):
    import scripts._geometry_references as references

    session = ReferenceSession(comparison, tmp_path / "references")
    session.save(0, POLYGONS, "approved")
    path = mask_path(session.output, "a/0.png")
    before_mask, before_metadata = path.read_bytes(), session.path.read_bytes()
    monkeypatch.setattr(references, "write_json", lambda *args: (_ for _ in ()).throw(OSError("metadata failed")))
    with pytest.raises(OSError, match="metadata failed"):
        session.save(0, [], "uncertain")
    assert path.read_bytes() == before_mask and session.path.read_bytes() == before_metadata
    assert session.data["images"][0]["status"] == "approved"


def test_reference_reuse_with_new_profile_and_unchanged_sources(comparison, tmp_path):
    session = ReferenceSession(comparison, tmp_path / "references")
    session.save(0, POLYGONS, "approved")
    new = tmp_path / "new_comparison"
    shutil.copytree(comparison, new)
    settings_path = new / "resolved_settings.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    name = "otsu_background_k61_hysteresis_mask"
    settings["profiles"][name] = deepcopy(settings["profiles"]["otsu_background_k61_mask"])
    settings["profiles"][name]["foreground"].update(enabled=True, method="otsu_background_hysteresis")
    settings_path.write_text(yaml.safe_dump(settings))
    shutil.copytree(new / "previews/otsu_mask", new / "previews" / name)
    measurements = pd.read_csv(new / "measurements.csv")
    extra = measurements.loc[measurements.profile == "otsu_mask"].assign(profile=name)
    pd.concat([measurements, extra]).to_csv(new / "measurements.csv", index=False)
    before = {path: digest(path) for path in session.output.rglob("*.*")}
    with pytest.raises(ValueError, match="provenance mismatch"):
        score_geometry_masks(new, session.output, tmp_path / "without_opt_in")
    frame = score_geometry_masks(new, session.output, tmp_path / "reused", reuse_references=True)
    assert len(frame) == 7
    assert (frame.iou_difference_vs_kernel61[frame.profile == name] == 0).all()
    assert (tmp_path / "reused/hysteresis_review_order.csv").exists()
    assert {path: digest(path) for path in before} == before
    settings["profiles"]["canny_baseline"]["resize"]["padding_color"] = "mean"
    settings_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="Reference reuse requires identical"):
        score_geometry_masks(new, session.output, tmp_path / "changed", reuse_references=True)
    assert not (tmp_path / "changed").exists()


@pytest.mark.parametrize("include_tuning", [False, True])
def test_quality_scoring_outputs_all_three_variants_and_preserves_references(comparison, tmp_path, include_tuning):
    session = ReferenceSession(comparison, tmp_path / "references")
    session.save(0, POLYGONS, "approved")
    before = {path: digest(path) for path in session.output.rglob("*.*")}
    new = tmp_path / "quality_comparison"
    shutil.copytree(comparison, new)
    settings_path = new / "resolved_settings.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    measurements = pd.read_csv(new / "measurements.csv")
    extra = []
    names = (*QUALITY_PROFILE_NAMES, *QUALITY_TUNING_PROFILES) if include_tuning else QUALITY_PROFILE_NAMES
    for name in names:
        settings["profiles"][name] = deepcopy(settings["profiles"]["otsu_background_k61_mask"])
        shutil.copytree(new / "previews/otsu_mask", new / "previews" / name)
        extra.append(measurements.loc[measurements.profile == "otsu_mask"].assign(profile=name))
    settings_path.write_text(yaml.safe_dump(settings))
    pd.concat([measurements, *extra]).to_csv(new / "measurements.csv", index=False)
    if include_tuning:
        assert cv2.imwrite(str(new / "previews/otsu_background_k61_bilateral_mask/a/0.png_image.png"), np.zeros((64, 64), np.uint8))
    output = tmp_path / "quality_scores"
    frame = score_geometry_masks(new, session.output, output, reuse_references=True)
    assert len(frame) == (12 if include_tuning else 9)
    quality = pd.read_csv(output / "quality_review_order.csv")
    assert set(quality.profile) == set(QUALITY_PROFILE_NAMES)
    if include_tuning:
        tuning = pd.read_csv(output / "quality_tuning_review_order.csv")
        assert set(tuning.profile) == set(QUALITY_TUNING_PROFILES)
        b2 = tuning.loc[tuning.profile == "otsu_background_k61_b2_mask"].iloc[0]
        assert b2.iou_difference_vs_kernel61 == 0
        assert b2.iou_difference_vs_previous_setting == 1
    else:
        assert (quality.iou_difference_vs_kernel61 == 0).all()
    assert cv2.imread(str(output / "error_comparisons/a/0.png_comparison.png")).shape[:2] == ((1950 if include_tuning else 1300), 1950)
    assert {path: digest(path) for path in before} == before
