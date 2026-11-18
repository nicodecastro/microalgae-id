"""Diagnostic-only Canny silhouettes and integration contracts."""

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from scripts._canny_masks import (
    ALL_CANNY_PROFILES, CANNY_MASK_PROFILES, CANNY_REFINEMENT_PROFILES,
    enclosed_mask, prepare_canny_mask,
)
from scripts import diagnose_geometry as diagnostic
from scripts.score_geometry_masks import score_geometry_masks
from test_geometry_diagnostics import saved_run
from test_geometry_references import comparison
from scripts._geometry_references import ReferenceSession


def test_enclosed_components_exclude_open_strokes_and_tiny_regions():
    edges = np.zeros((100, 100), np.uint8)
    cv2.rectangle(edges, (5, 5), (30, 30), 255, 1)
    cv2.rectangle(edges, (50, 50), (80, 80), 255, 1)
    cv2.line(edges, (40, 5), (40, 40), 255)
    cv2.rectangle(edges, (90, 90), (92, 92), 255, 1)
    mask = enclosed_mask(edges)
    assert mask[15, 15] and mask[60, 60]
    assert not mask[20, 40] and not mask[91, 91]
    assert cv2.connectedComponents(mask.astype(np.uint8))[0] == 3


def test_touching_objects_are_not_split_and_border_is_not_forced_closed():
    edges = np.zeros((80, 80), np.uint8)
    cv2.circle(edges, (25, 40), 15, 255, 1)
    cv2.circle(edges, (55, 40), 15, 255, 1)
    assert cv2.connectedComponents(enclosed_mask(edges).astype(np.uint8))[0] == 2
    edges[:] = 0
    edges[10, :40] = 255
    edges[50, :40] = 255
    edges[10:51, 40] = 255
    assert not enclosed_mask(edges).any()


def test_small_gap_can_close_but_open_stroke_has_no_interior():
    edges = np.zeros((50, 50), np.uint8)
    cv2.rectangle(edges, (10, 10), (40, 40), 255, 2)
    edges[9:12, 25] = 0
    assert not enclosed_mask(edges)[25, 25]
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    assert enclosed_mask(closed)[25, 25]
    edges[:] = 0
    cv2.line(edges, (10, 10), (40, 40), 255)
    assert not enclosed_mask(edges).any()


def test_content_only_and_intermediates_exclude_padding():
    canvas = np.zeros((100, 120, 3), np.uint8)
    content = np.zeros((100, 120), bool)
    content[20:80, 10:110] = True
    canvas[content] = 220
    cv2.ellipse(canvas, (35, 50), (12, 15), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(canvas, (80, 50), (12, 15), 0, 0, 360, (20, 20, 20), -1)
    first = prepare_canny_mask(canvas, content, [50, 100])
    canvas[~content] = 255
    second = prepare_canny_mask(canvas, content, [50, 100])
    np.testing.assert_array_equal(first.image, second.image)
    assert first.mask[50, 35] and first.mask[50, 80]
    assert not first.mask[~content].any()
    assert set(first.intermediates) == {"blurred", "edges", "closed_edges"}
    for key, image in first.intermediates.items():
        assert not image[~content].any()
        np.testing.assert_array_equal(image, second.intermediates[key])
    canvas[content] = 180
    assert not prepare_canny_mask(canvas, content, [50, 100]).mask.any()


def test_profiles_are_opt_in_and_composable(saved_run):
    _, old_selection, old = diagnostic._load_selection(saved_run)
    _, selection, profiles = diagnostic._load_selection(saved_run, include_canny_masks=True)
    pd.testing.assert_frame_equal(old_selection, selection)
    assert {key: profiles[key] for key in old} == old
    assert list(profiles)[-3:] == list(CANNY_MASK_PROFILES)
    _, _, combined = diagnostic._load_selection(saved_run, include_canny_masks=True,
        include_hysteresis=True, include_quality=True, include_quality_tuning=True)
    assert set(profiles).issubset(combined)


@pytest.mark.parametrize("thresholds", [[50, 100], [20, 40]])
def test_rectangular_closing_repairs_a_gap_without_changing_edges(monkeypatch, thresholds):
    edges = np.zeros((60, 60), np.uint8)
    cv2.rectangle(edges, (10, 10), (45, 45), 255, 1)
    edges[10, 25] = 0
    monkeypatch.setattr(cv2, "Canny", lambda *args, **kwargs: edges.copy())
    canvas = np.full((80, 80, 3), 180, np.uint8)
    content = np.zeros((80, 80), bool)
    content[10:70, 10:70] = True
    ellipse = prepare_canny_mask(canvas, content, thresholds)
    rectangle = prepare_canny_mask(canvas, content, thresholds, closing_shape="rectangle")
    assert not ellipse.mask[35, 35] and rectangle.mask[35, 35]
    np.testing.assert_array_equal(ellipse.intermediates["edges"], rectangle.intermediates["edges"])
    np.testing.assert_array_equal(ellipse.intermediates["blurred"], rectangle.intermediates["blurred"])
    for stage in [rectangle.image, *rectangle.intermediates.values()]:
        assert not stage[~content].any()
    with pytest.raises(ValueError, match="closing_shape"):
        prepare_canny_mask(canvas, content, [50, 100], closing_shape="unknown")


def test_refinements_keep_original_settings_and_artifacts(saved_run, tmp_path, monkeypatch):
    _, original_selection, original = diagnostic._load_selection(saved_run, include_canny_masks=True)
    _, selection, profiles = diagnostic._load_selection(saved_run, include_canny_refinements=True)
    pd.testing.assert_frame_equal(original_selection, selection)
    assert {name: profiles[name] for name in original} == original
    assert len(profiles) == 13 and list(profiles)[-len(CANNY_REFINEMENT_PROFILES):] == list(CANNY_REFINEMENT_PROFILES)
    _, _, combined = diagnostic._load_selection(saved_run, include_canny_refinements=True,
        include_hysteresis=True, include_quality=True, include_quality_tuning=True)
    assert set(profiles).issubset(combined)
    monkeypatch.setattr(diagnostic, "_save_comparison", lambda *args: None)
    output = tmp_path / "refinement"
    result = diagnostic.diagnose_geometry(saved_run, output, include_canny_refinements=True)
    assert result["processing_errors"] == 0
    assert result["review_rows"] == result["selected_images"] * 13
    snapshot = yaml.safe_load((output / "resolved_settings.yaml").read_text())
    for name, parameters in CANNY_REFINEMENT_PROFILES.items():
        assert snapshot["profiles"][name]["thresholds"] == parameters["thresholds"]
        assert snapshot["profiles"][name]["closing_kernel"] == "3x3 " + parameters["closing_shape"]
        for identifier in selection.image_id:
            for stage in ("image", "mask", "blurred", "edges", "closed_edges"):
                assert (output / "previews" / name / (identifier + f"_{stage}.png")).exists()
    assert pd.read_csv(output / "review.csv").review_status.eq("unreviewed").all()


def test_diagnostic_outputs_and_flag_off_pixels(saved_run, tmp_path):
    original, candidate = tmp_path / "original", tmp_path / "candidate"
    diagnostic.diagnose_geometry(saved_run, original)
    result = diagnostic.diagnose_geometry(saved_run, candidate, include_canny_masks=True)
    assert result["processing_errors"] == 0
    assert result["review_rows"] == result["selected_images"] * 9
    old = pd.read_csv(original / "measurements.csv")
    new = pd.read_csv(candidate / "measurements.csv")
    pd.testing.assert_frame_equal(old, new[new.profile.isin(diagnostic.PROFILE_NAMES)].reset_index(drop=True))
    for path in (original / "previews").rglob("*.png"):
        assert path.read_bytes() == (candidate / path.relative_to(original)).read_bytes()
    settings = yaml.safe_load((candidate / "resolved_settings.yaml").read_text())
    assert settings["profiles"]["canny_50_100_mask"]["thresholds"] == [50, 100]
    assert settings["canny_masks"]["require_newly_filled_interior"]
    assert pd.read_csv(candidate / "review.csv").review_status.eq("unreviewed").all()


@pytest.mark.parametrize("refinements", [False, True])
def test_scoring_labels_ranking_and_review_handoff(comparison, tmp_path, monkeypatch, refinements):
    session = ReferenceSession(comparison, tmp_path / "references")
    polygons = [{"mode": "add", "vertices": [[15, 22], [35, 22], [35, 38], [15, 38]]}]
    for index in range(len(session.data["images"])):
        session.save(index, polygons, "approved")
    settings_path = comparison / "resolved_settings.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    measurements = pd.read_csv(comparison / "measurements.csv")
    extra = []
    candidates = {name: {"thresholds": thresholds} for name, thresholds in CANNY_MASK_PROFILES.items()}
    if refinements:
        candidates.update(CANNY_REFINEMENT_PROFILES)
    for profile, parameters in candidates.items():
        settings["profiles"][profile] = parameters
        for identifier in measurements.image_id.unique():
            source = comparison / "previews" / "otsu_mask" / (identifier + "_image.png")
            target = comparison / "previews" / profile / (identifier + "_image.png")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            extra.append({"image_id": identifier, "profile": profile, "extraction_status": "measured"})
    # References retain original provenance; reuse validates the unchanged baseline and inventory.
    other = tmp_path / "new_comparison"
    import shutil
    shutil.copytree(comparison, other)
    (other / "resolved_settings.yaml").write_text(yaml.safe_dump(settings))
    pd.concat([measurements, pd.DataFrame(extra)]).to_csv(other / "measurements.csv", index=False)
    selection = pd.read_csv(other / "selection.csv")
    pd.DataFrame([{"image_id": row.image_id, "profile": profile, "review_status": "unreviewed", "notes": ""}
                  for row in selection.itertuples() for profile in settings["profiles"]]).to_csv(other / "review.csv", index=False)
    output = tmp_path / "scores"
    scores = score_geometry_masks(other, session.output, output, reuse_references=True)
    assert len(scores) == 12 * (6 + len(candidates))
    assert scores.loc[scores.profile.isin(candidates), "measurement_type"].eq("filled_mask_overlap").all()
    assert scores.loc[scores.profile == "canny_baseline", "measurement_type"].eq("edge_input_overlap").all()
    ranking = pd.read_csv(output / "canny_ranking.csv")
    assert ranking.profile.tolist() == sorted(candidates)
    assert len(pd.read_csv(output / "canny_selected_review.csv")) == 12
    assert "recall_difference_vs_kernel61" in pd.read_csv(output / "canny_review_order.csv")
    assert pd.read_csv(output / "canny_review_order.csv").iou_difference_vs_canny50_100.eq(0).all()

    import json
    status = json.loads((output / "canny_review_status.json").read_text())
    assert not status["review_complete"] and not status["adoption_approved"]
    reviews = pd.read_csv(other / "review.csv", keep_default_na=False)
    reviews.loc[reviews.profile == ranking.iloc[0].profile, "review_status"] = "acceptable"
    reviews.to_csv(other / "review.csv", index=False)
    monkeypatch.setattr("scripts.score_geometry_masks.save_error_comparison", lambda *args: None)
    reviewed_output = tmp_path / "reviewed_scores"
    score_geometry_masks(other, session.output, reviewed_output, reuse_references=True)
    status = json.loads((reviewed_output / "canny_review_status.json").read_text())
    assert status["review_complete"] and status["acceptable_count"] == 12
    assert status["recommendation"] == "unpromising under the exploratory gate"
    assert not status["adoption_approved"]
