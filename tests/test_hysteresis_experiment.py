"""Five-profile experiment, historical references and equal-class weighting."""

from copy import deepcopy
import json
import shutil

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from microalgae.imaging.preprocessing.configured import prepare_feature_input, validate_preprocessing
from microalgae.imaging.preprocessing.geometry_foreground import geometry_foreground_mask
from scripts import diagnose_geometry as diagnostic
from scripts import score_geometry_masks as scoring
from scripts._geometry_references import ReferenceSession, digest, load_comparison
from test_geometry_diagnostics import saved_run
from test_geometry_references import comparison, POLYGONS


def test_exact_profiles_and_source_baseline_preserved(saved_run):
    _, old_selection, old = diagnostic._load_selection(saved_run, include_hysteresis=True)
    _, selection, profiles = diagnostic._load_selection(saved_run, hysteresis_experiment=True)
    pd.testing.assert_frame_equal(selection, old_selection)
    assert tuple(profiles) == diagnostic.HYSTERESIS_EXPERIMENT_PROFILES
    assert profiles[diagnostic.HYSTERESIS_CONTROL] == old[diagnostic.HYSTERESIS_CONTROL]
    assert [row["foreground"]["hysteresis"]["low_ratio"] for row in profiles.values()] == [0.5, 0.65, 0.8, 0.5, 0.5]
    hb = profiles[diagnostic.HYSTERESIS_EXPERIMENT_PROFILES[3]]
    hc = profiles[diagnostic.HYSTERESIS_EXPERIMENT_PROFILES[4]]
    assert hb["filter"]["enabled"] and hb["filter"]["bilateral"]["params"] == [5, 15, 3]
    assert not hb["foreground"]["hysteresis"]["clahe"]["enabled"]
    assert not hc["filter"]["enabled"] and hc["foreground"]["hysteresis"]["clahe"]["enabled"]
    assert all(row["foreground"]["background"]["kernel_size"] == 61 for row in profiles.values())


@pytest.mark.parametrize("flag", ["include_hysteresis", "include_quality", "include_quality_tuning",
                                  "include_canny_masks", "include_canny_refinements"])
def test_experiment_flags_exclusive_before_writes(tmp_path, flag):
    with pytest.raises(ValueError, match="cannot be combined"):
        diagnostic.diagnose_geometry(tmp_path / "missing", tmp_path / "output",
                                    hysteresis_experiment=True, **{flag: True})
    assert not (tmp_path / "output").exists()


def test_ratio_growth_default_compatibility_and_otsu_limit():
    image = np.full((100, 150), 200, np.uint8)
    image[30:65, 20:55] = 180
    image[32:57, 22:47] = 140
    image[30:65, 95:125] = 180
    content = np.ones_like(image, bool)
    masks = [geometry_foreground_mask(image, content, method="otsu_background_hysteresis", hysteresis_low_ratio=r)
             for r in (0.5, 0.65, 0.8, 1)]
    np.testing.assert_array_equal(masks[0], geometry_foreground_mask(image, content, method="otsu_background_hysteresis"))
    np.testing.assert_array_equal(masks[-1], geometry_foreground_mask(image, content, method="otsu_background"))
    for larger, smaller in zip(masks, masks[1:]):
        assert not np.any(smaller & ~larger)
    assert masks[0][60, 50] and masks[1][60, 50] and not masks[2][60, 50]
    assert all(not mask[45, 110] for mask in masks)


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1, True, float("nan"), float("inf"), "0.5"])
def test_invalid_ratios_in_direct_and_configured_paths(ratio):
    with pytest.raises(ValueError, match="low_ratio"):
        geometry_foreground_mask(np.zeros((20, 20), np.uint8), np.ones((20, 20), bool),
                                 method="otsu_background_hysteresis", hysteresis_low_ratio=ratio)
    with pytest.raises(ValueError, match="low_ratio"):
        validate_preprocessing({"foreground": {"enabled": True, "method": "otsu_background_hysteresis",
            "hysteresis": {"low_ratio": ratio}}}, "geom")


@pytest.mark.parametrize("updates", [{"clip_limit": 0}, {"clip_limit": True}, {"clip_limit": float("inf")},
                                      {"tile_grid_size": [0, 4]}, {"tile_grid_size": [True, 4]},
                                      {"tile_grid_size": [2.5, 4]}, {"tile_grid_size": [4]}])
def test_invalid_hysteresis_clahe(updates):
    with pytest.raises(ValueError, match="CLAHE"):
        validate_preprocessing({"foreground": {"enabled": True, "method": "otsu_background_hysteresis",
            "hysteresis": {"clahe": {"enabled": True, **updates}}}}, "geom")


def test_hc_placement_padding_and_constant_inputs():
    roi = np.random.default_rng(42).integers(100, 200, (60, 90), dtype=np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4)).apply(roi)
    expected = geometry_foreground_mask(enhanced, np.ones_like(roi, bool), method="otsu_background_hysteresis")
    settings = validate_preprocessing({"resize": {"enabled": False}, "foreground": {
        "enabled": True, "method": "otsu_background_hysteresis", "hysteresis": {"clahe": {"enabled": True}}}}, "geom")
    prepared = prepare_feature_input(roi, settings, "geom")
    np.testing.assert_array_equal(prepared.mask, expected)
    np.testing.assert_array_equal(prepared.intermediates["contrast"], enhanced)
    content = np.zeros((80, 110), bool)
    content[10:70, 10:100] = True
    for padding in (0, 255):
        image = np.full(content.shape, padding, np.uint8)
        image[content] = roi.ravel()
        stages = {}
        mask = geometry_foreground_mask(image, content, method="otsu_background_hysteresis",
            hysteresis_clahe_params={"clip_limit": 1.5, "tile_grid_size": [4, 4]}, intermediates=stages)
        np.testing.assert_array_equal(mask[content].reshape(roi.shape), expected)
        assert not mask[~content].any() and all(not v[~content].any() for v in stages.values())
    for ratio in (0.5, 0.65, 0.8):
        assert not geometry_foreground_mask(np.full((30, 40), 100, np.uint8), np.ones((30, 40), bool),
            method="otsu_background_hysteresis", hysteresis_low_ratio=ratio).any()


def test_equal_class_weighting_and_unrounded_ties():
    rows = []
    for profile, values in ((diagnostic.HYSTERESIS_CONTROL, [0.9, 0.1, 0.1]), ("balanced", [0.5, 0.5, 0.5]),
                            ("tied_z", [0.5, 0.5, 0.5]), ("tied_a", [0.5, 0.5, 0.5])):
        for label, n, value in zip("abc", (8, 1, 1), values):
            for i in range(n):
                rows.append({"image_id": f"{label}/{i}", "Class": label, "profile": profile,
                             "iou": value, "dice": value, "precision": value, "recall": value})
    frame = pd.DataFrame(rows)
    classes, ranking = scoring.hysteresis_ranking(frame)
    assert ranking.iloc[0].profile == "balanced"
    assert ranking.profile.tolist()[:3] == ["balanced", "tied_a", "tied_z"]
    control = ranking.set_index("profile").loc[diagnostic.HYSTERESIS_CONTROL]
    assert control.macro_iou == pytest.approx(1.1 / 3)
    assert control.image_weighted_iou == pytest.approx(0.74)
    frame.loc[frame.profile == "tied_z", "dice"] = 0.8
    assert scoring.hysteresis_ranking(frame)[1].iloc[0].profile == "tied_z"
    frame.loc[frame.profile == "tied_a", "iou"] += 1e-8
    assert scoring.hysteresis_ranking(frame)[1].iloc[0].profile == "tied_a"
    with pytest.raises(ValueError, match="same images"):
        scoring.hysteresis_ranking(frame.iloc[1:])


def test_historical_77_references_reused_without_legacy_profiles(comparison, tmp_path, monkeypatch):
    source_settings = yaml.safe_load((comparison / "resolved_settings.yaml").read_text())
    for settings in source_settings["profiles"].values():
        settings["foreground"].pop("hysteresis", None)
    (comparison / "resolved_settings.yaml").write_text(yaml.safe_dump(source_settings))
    inventory = json.loads((comparison / "source_images.json").read_text())
    measurements = pd.read_csv(comparison / "measurements.csv")
    additions = []
    for i in range(12, 77):
        identifier = f"a/{i}.png"
        shutil.copyfile(tmp_path / "raw/a/0.png", tmp_path / "raw" / identifier)
        inventory.append({**inventory[0], "image_id": identifier})
        for profile in source_settings["profiles"]:
            shutil.copyfile(comparison / "previews" / profile / "a/0.png_image.png",
                            comparison / "previews" / profile / (identifier + "_image.png"))
        additions.append(measurements.loc[measurements.image_id == "a/0.png"].assign(image_id=identifier))
    selection = pd.DataFrame({"image_id": [f"a/{i}.png" for i in range(77)],
                              "Class": ["a"] * 60 + ["b"] * 10 + ["c"] * 7,
                              "baseline_status": ["failed"] * 77})
    selection.to_csv(comparison / "selection.csv", index=False)
    selection.assign(selected=True).to_csv(comparison / "reference_selection.csv", index=False)
    (comparison / "source_images.json").write_text(json.dumps(inventory))
    pd.concat([measurements, *additions]).to_csv(comparison / "measurements.csv", index=False)
    session = ReferenceSession(comparison, tmp_path / "references", select_all=True)
    for i in range(77):
        session.save(i, POLYGONS, "approved")
    before = {p: digest(p) for p in session.output.rglob("*.*")}
    new = tmp_path / "five_profiles"
    shutil.copytree(comparison, new)
    settings = {**source_settings, "experiment": "hysteresis_refinement",
                "baseline_preprocessing": deepcopy(source_settings["profiles"]["canny_baseline"]),
                "profiles": {p: deepcopy(source_settings["profiles"]["otsu_mask"])
                             for p in diagnostic.HYSTERESIS_EXPERIMENT_PROFILES}}
    (new / "resolved_settings.yaml").write_text(yaml.safe_dump(settings))
    old = pd.read_csv(new / "measurements.csv")
    pd.concat([old.loc[old.profile == "otsu_mask"].assign(profile=p)
               for p in settings["profiles"]]).to_csv(new / "measurements.csv", index=False)
    for profile in settings["profiles"]:
        shutil.copytree(new / "previews/otsu_mask", new / "previews" / profile)
    assert set(load_comparison(new)[1]["profiles"]) == set(diagnostic.HYSTERESIS_EXPERIMENT_PROFILES)
    original = scoring.save_error_comparison
    def one_gallery(path, identifier, canvas, reference, views):
        assert len(views) == 5
        if identifier == "a/0.png":
            original(path, identifier, canvas, reference, views)
    monkeypatch.setattr(scoring, "save_error_comparison", one_gallery)
    output = tmp_path / "scores"
    frame = scoring.score_geometry_masks(new, session.output, output, reuse_references=True)
    assert len(frame) == 385
    assert not any("dark_otsu" in c or "kernel61" in c for c in frame)
    assert (frame.iou_difference_vs_hysteresis == 0).all()
    assert len(pd.read_csv(output / "summary_by_class.csv")) == 15
    assert len(pd.read_csv(output / "paired_vs_hysteresis.csv")) == 308
    assert (pd.read_csv(output / "profile_ranking.csv").macro_iou == 1).all()
    assert json.loads((output / "scoring_settings.json").read_text())["scored_count"] == 77
    assert cv2.imread(str(output / "error_comparisons/a/0.png_comparison.png")).shape[:2] == (1300, 1950)
    assert {p: digest(p) for p in before} == before
    session.save(76, [], "uncertain")
    with pytest.raises(ValueError, match="77 references approved"):
        scoring.score_geometry_masks(new, session.output, tmp_path / "incomplete", reuse_references=True)
    assert not (tmp_path / "incomplete").exists()
