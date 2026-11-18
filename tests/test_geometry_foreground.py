"""Opt-in component masks preserve legacy geometry and exclude artificial padding."""

from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import pytest

from microalgae.imaging.features import geometric_features as geom
from microalgae.imaging.preprocessing import configured as prep
from microalgae.imaging.preprocessing.geometry_foreground import geometry_foreground_mask
from microalgae.imaging.preprocessing.resize import resize_with_padding
from training.config import load_config
from training.extraction import extract_configured_features


def foreground_settings(method="otsu", **extra):
    return prep.validate_preprocessing({
        "resize": {"target_size": [180, 180]},
        "foreground": {"enabled": True, "method": method, **extra},
    }, "geom")


def objects():
    image = np.full((70, 140), 210, dtype=np.uint8)
    cv2.ellipse(image, (35, 35), (17, 11), 20, 0, 360, 30, -1)
    cv2.circle(image, (100, 35), 13, 30, -1)
    return image


@pytest.mark.parametrize("method", ["otsu", "otsu_bright", "adaptive", "otsu_background"])
@pytest.mark.parametrize("denoise", [False, True])
def test_padding_does_not_affect_masks_or_descriptors(method, denoise):
    image = cv2.cvtColor(objects(), cv2.COLOR_GRAY2BGR)
    if method == "otsu_bright":
        image = 255 - image
    original = image.copy()
    settings = foreground_settings(method)
    settings["filter"]["enabled"] = denoise
    settings["filter"]["bilateral"]["params"] = [5, 75, 75]
    outputs = []
    for color in ["black", "mean"]:
        settings["resize"]["padding_color"] = color
        result = prep.prepare_feature_input(image, settings, "geom")
        _, content = resize_with_padding(image, (180, 180), return_content_mask=True)
        assert result.mask.dtype == bool and not result.mask[~content].any()
        np.testing.assert_array_equal(result.image, result.mask.astype(np.uint8) * 255)
        outputs.append(result)
    np.testing.assert_array_equal(outputs[0].mask, outputs[1].mask)
    np.testing.assert_allclose(geom.extract_geometric_features(outputs[0].image), geom.extract_geometric_features(outputs[1].image))
    np.testing.assert_array_equal(image, original)


def test_bright_otsu_selects_bright_object_instead_of_dark_background():
    image = np.full((80, 120), 30, dtype=np.uint8)
    cv2.ellipse(image, (60, 40), (23, 12), 20, 0, 360, 220, -1)
    content = np.ones_like(image, dtype=bool)
    bright = geometry_foreground_mask(image, content, method="otsu_bright")
    dark = geometry_foreground_mask(image, content)
    assert bright[40, 60] and not bright[0, 0]
    assert 0.05 < bright.mean() < 0.2
    assert dark.all()  # Dark background encloses the bright object; hole filling fills it.
    settings = foreground_settings("otsu_bright")
    settings["resize"]["enabled"] = False
    prepared = prep.prepare_feature_input(image, settings, "geom")
    np.testing.assert_array_equal(prepared.mask, bright)
    assert geom.extract_geometric_features(prepared.image) is not None


def test_resize_mask_is_opt_in_and_marks_exact_real_pixels():
    image = np.arange(30, dtype=np.uint8).reshape(5, 6)
    old = resize_with_padding(image, (20, 20), 99)
    canvas, content = resize_with_padding(image, (20, 20), 99, return_content_mask=True)
    np.testing.assert_array_equal(canvas, old)
    assert content.dtype == bool
    assert content.sum() == 16 * 20
    assert content[2:18].all() and not content[:2].any() and not content[18:].any()
    assert np.all(canvas[~content] == 99)
    with pytest.raises(ValueError, match="Boolean"):
        resize_with_padding(image, return_content_mask="yes")


def test_masks_fill_holes_remove_small_components_and_keep_multiple_objects():
    image = objects()
    cv2.circle(image, (35, 35), 4, 210, -1)
    image[4:7, 4:7] = 30
    result = geometry_foreground_mask(image, np.ones_like(image, dtype=bool))
    assert result[35, 35]  # Hole filled.
    assert not result[4:7, 4:7].any()
    assert cv2.connectedComponents(result.astype(np.uint8), connectivity=8)[0] - 1 == 2
    contours, _ = cv2.findContours(result.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = geom.extract_geometric_features(result.astype(np.uint8) * 255)
    assert features[0] == pytest.approx(np.mean([cv2.contourArea(contour) for contour in contours]))


def test_border_components_are_kept_not_silently_removed():
    image = np.full((70, 100), 210, dtype=np.uint8)
    cv2.circle(image, (0, 35), 14, 25, -1)
    result = geometry_foreground_mask(image, np.ones_like(image, dtype=bool))
    assert result[:, 0].any()
    assert geom.extract_geometric_features(result.astype(np.uint8) * 255) is not None


@pytest.mark.parametrize("value", [0, 100, 255])
@pytest.mark.parametrize("method", ["otsu", "otsu_bright", "adaptive", "otsu_background"])
def test_constant_images_have_empty_masks(value, method):
    image = np.full((40, 60), value, dtype=np.uint8)
    result = prep.prepare_feature_input(image, foreground_settings(method), "geom")
    assert not result.mask.any()
    assert geom.extract_geometric_features(result.image) is None


def test_default_and_non_geometry_paths_are_unchanged():
    config = load_config()
    original = deepcopy(config)
    settings = prep.resolve_preprocessing(config, "geom")
    assert settings["foreground"] == {
        "enabled": False, "method": "otsu", "min_component_area": 25,
        "adaptive": {"block_size": 31, "c": 3},
        "background": {"kernel_size": 61},
        "hysteresis": {"low_ratio": 0.5,
                       "clahe": {"enabled": False, "clip_limit": 1.5, "tile_grid_size": [4, 4]}},
    }
    image = cv2.cvtColor(objects(), cv2.COLOR_GRAY2BGR)
    result = prep.prepare_feature_input(image, settings, "geom")
    gray = cv2.cvtColor(resize_with_padding(image, (256, 256)), cv2.COLOR_BGR2GRAY)
    np.testing.assert_array_equal(result.image, cv2.Canny(gray, 150, 175))
    assert result.mask is None
    for family in ("efd", "gabor", "hu", "lbp", "glcm", "pc"):
        assert "foreground" not in prep.resolve_preprocessing(config, family)
        with pytest.raises(ValueError, match="Unknown setting"):
            prep.validate_preprocessing({"foreground": {"enabled": False}}, family)
    assert config == original


@pytest.mark.parametrize("change", [
    {"enabled": "yes"}, {"method": "invalid"}, {"min_component_area": 0},
    {"min_component_area": True}, {"min_component_area": 2.5},
    {"method": "adaptive", "adaptive": {"block_size": 4}},
    {"method": "adaptive", "adaptive": {"c": float("nan")}},
])
def test_invalid_foreground_settings_raise_before_processing(change):
    with pytest.raises(ValueError, match="foreground"):
        prep.validate_preprocessing({"foreground": {"enabled": True, **change}}, "geom")


@pytest.mark.parametrize("stage", ["threshold", "segment"])
def test_threshold_and_edge_conflicts_are_rejected(stage):
    with pytest.raises(ValueError, match="disabled threshold/edge"):
        prep.validate_preprocessing({"foreground": {"enabled": True}, stage: {"enabled": True}}, "geom")


def test_direct_mask_validation():
    image = objects()
    content = np.ones_like(image, dtype=bool)
    with pytest.raises(ValueError, match="uint8"):
        geometry_foreground_mask(image.astype(float), content)
    with pytest.raises(ValueError, match="Boolean"):
        geometry_foreground_mask(image, content.astype(np.uint8))
    content[20, 20] = False
    with pytest.raises(ValueError, match="rectangular"):
        geometry_foreground_mask(image, content)
    with pytest.raises(ValueError, match="method"):
        geometry_foreground_mask(image, np.ones_like(image, dtype=bool), method="bad")


def test_mask_stages_run_once_and_denoising_sees_no_padding(monkeypatch):
    from microalgae.imaging.preprocessing import geometry_foreground as foreground

    calls = []
    resize = prep.resize_with_padding
    bilateral = foreground.bilateral_filter
    mask_function = prep.geometry_foreground_mask

    def resized(*args, **kwargs):
        calls.append("resize")
        return resize(*args, **kwargs)

    def denoised(image, *args):
        calls.append("denoise")
        assert image.shape == (90, 180)
        return bilateral(image, *args)

    def masked(*args, **kwargs):
        calls.append("mask")
        return mask_function(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Foreground mode must not run legacy edges or Sanchez segmentation")

    monkeypatch.setattr(prep, "resize_with_padding", resized)
    monkeypatch.setattr(foreground, "bilateral_filter", denoised)
    monkeypatch.setattr(prep, "geometry_foreground_mask", masked)
    for name in ["canny", "sobel", "segment_image"]:
        monkeypatch.setattr(prep, name, forbidden)
    settings = foreground_settings()
    settings["filter"]["enabled"] = True
    prep.prepare_feature_input(objects(), settings, "geom")
    assert calls == ["resize", "mask", "denoise"]


def test_configured_mask_export_matches_direct_calculation_and_previews(tmp_path):
    root = tmp_path / "raw"
    (root / "a").mkdir(parents=True)
    image = cv2.cvtColor(objects(), cv2.COLOR_GRAY2BGR)
    assert cv2.imwrite(str(root / "a" / "image.png"), image)
    settings = foreground_settings()
    prepared = prep.prepare_feature_input(image, settings, "geom")
    frame = geom.extract_features(root, ["a"], tmp_path / "features", True, standardize=False, preprocessing=settings)
    np.testing.assert_allclose(frame[geom.FEATURE_NAMES].iloc[0], geom.extract_geometric_features(prepared.image))
    assert frame.columns.tolist() == [*geom.FEATURE_NAMES, "image_id", "Class"]
    assert frame.image_id.tolist() == ["a/image.png"]
    for suffix in ["image", "mask"]:
        np.testing.assert_array_equal(cv2.imread(str(tmp_path / "features" / "previews" / "geom" / "a" / f"image.png_{suffix}.png"), 0), prepared.image)
    assert pd.read_csv(tmp_path / "features" / "Geometric_shape_features_failures.csv").empty


def test_empty_masks_still_fail_strict_auditing_and_snapshot_records_variant(tmp_path):
    root = tmp_path / "raw"
    (root / "a").mkdir(parents=True)
    assert cv2.imwrite(str(root / "a" / "good.png"), objects())
    assert cv2.imwrite(str(root / "a" / "blank.png"), np.zeros((40, 40), dtype=np.uint8))
    config = load_config()
    config["dataset"].update(raw_dir=str(root), classes=["a"])
    config["features"]["feature_list"] = ["geom"]
    config["preprocessing"]["overrides"]["geom"] = {"foreground": {"enabled": True}}
    with pytest.warns(UserWarning, match="Skipping"), pytest.raises(ValueError, match="audit failed"):
        extract_configured_features(config, tmp_path / "run", standardize=False)
    issues = pd.read_csv(tmp_path / "run" / "feature_table_audit.csv")
    assert {"missing_image", "extraction_failure"} <= set(issues.issue)
    import yaml
    snapshot = yaml.safe_load((tmp_path / "run" / "extraction_settings.yaml").read_text())
    assert snapshot["resolved_preprocessing"]["geom"]["foreground"]["enabled"] is True
