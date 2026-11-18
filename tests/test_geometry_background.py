"""Background correction matches explicit operations without changing old methods."""

import cv2
import numpy as np
import pytest
from scipy.ndimage import binary_fill_holes

from microalgae.imaging.preprocessing.configured import prepare_feature_input, validate_preprocessing, save_prepared_input
from microalgae.imaging.preprocessing.geometry_foreground import geometry_foreground_mask


@pytest.mark.parametrize("kernel", [31, 61])
def test_background_correction_matches_independent_calculation_and_known_objects(kernel, tmp_path):
    image = np.tile(np.linspace(150, 240, 150).astype(np.uint8), (100, 1))
    truth = np.zeros_like(image)
    cv2.circle(truth, (0, 55), 9, 1, -1)  # Clipped object retained.
    cv2.ellipse(truth, (65, 50), (9, 15), 20, 0, 360, 1, -1)
    cv2.circle(truth, (115, 50), 9, 1, -1)
    image = cv2.subtract(image, truth * 90)
    smoothed = cv2.GaussianBlur(image, (3, 3), 0)
    background = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    residual = np.maximum(background.astype(int) - smoothed.astype(int), 0).astype(np.uint8)
    _, expected = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    expected = cv2.morphologyEx(expected, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    expected = binary_fill_holes(expected != 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(expected, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= 25
    keep[0] = False
    expected = keep[labels]
    settings = validate_preprocessing({"resize": {"enabled": False}, "foreground": {
        "enabled": True, "method": "otsu_background", "background": {"kernel_size": kernel}}}, "geom")
    prepared = prepare_feature_input(image, settings, "geom")
    np.testing.assert_array_equal(prepared.mask, expected)
    np.testing.assert_array_equal(prepared.intermediates["background"], background)
    np.testing.assert_array_equal(prepared.intermediates["residual"], residual)
    assert prepared.mask[55, 0] and prepared.mask[50, 65] and prepared.mask[50, 115]
    assert np.count_nonzero(prepared.mask & (truth != 0)) / np.count_nonzero(prepared.mask | (truth != 0)) > 0.8
    save_prepared_input(prepared, tmp_path, "profile", "a.png")
    for name in ("background", "residual"):
        np.testing.assert_array_equal(cv2.imread(str(tmp_path / "previews" / "profile" / f"a.png_{name}.png"), 0), prepared.intermediates[name])


@pytest.mark.parametrize("kernel", [True, 0, 2, 4, 3.5, float("nan")])
def test_invalid_background_kernel_rejected_in_config_and_direct_helper(kernel):
    with pytest.raises(ValueError, match="kernel_size"):
        validate_preprocessing({"foreground": {"enabled": True, "method": "otsu_background",
                                             "background": {"kernel_size": kernel}}}, "geom")
    with pytest.raises(ValueError, match="kernel_size"):
        geometry_foreground_mask(np.zeros((10, 10), np.uint8), np.ones((10, 10), bool),
                                 method="otsu_background", background_kernel_size=kernel)


def test_constant_residual_empty_and_intermediates_never_include_padding():
    image = np.full((40, 60), 130, np.uint8)
    content = np.zeros_like(image, bool)
    content[5:35, 10:50] = True
    stages = {}
    mask = geometry_foreground_mask(image, content, method="otsu_background", intermediates=stages)
    assert not mask.any()
    assert not stages["residual"].any()
    assert (stages["background"][content] == 130).all()
    assert not stages["background"][~content].any()


def test_existing_otsu_bright_and_adaptive_outputs_unchanged():
    image = np.full((80, 100), 180, np.uint8)
    cv2.circle(image, (50, 40), 16, 20, -1)
    smoothed = cv2.GaussianBlur(image, (3, 3), 0)
    for method in ("otsu", "otsu_bright", "adaptive"):
        if method == "adaptive":
            binary = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
        else:
            _, binary = cv2.threshold(smoothed, 0, 255, (cv2.THRESH_BINARY_INV if method == "otsu" else cv2.THRESH_BINARY) | cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        binary = binary_fill_holes(binary != 0).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= 25
        keep[0] = False
        np.testing.assert_array_equal(geometry_foreground_mask(image, np.ones_like(image, bool), method=method), keep[labels])


def test_hysteresis_recovers_connected_weak_region_but_not_isolated_weak_object():
    image = np.full((100, 150), 200, np.uint8)
    image[30:65, 20:55] = 180  # Weak silhouette attached to a dark seed.
    image[32:57, 22:47] = 140
    image[30:65, 95:125] = 180  # Equally weak, but disconnected from any seed.
    settings = validate_preprocessing({"resize": {"enabled": False}, "foreground": {
        "enabled": True, "method": "otsu_background_hysteresis"}}, "geom")
    prepared = prepare_feature_input(image, settings, "geom")
    assert prepared.mask[60, 50] and prepared.mask[45, 35]
    assert not prepared.mask[45, 110]
    assert prepared.intermediates["growth_allowed"][45, 110] == 255
    assert prepared.intermediates["seeds"][45, 110] == 0
    baseline = geometry_foreground_mask(image, np.ones_like(image, bool), method="otsu_background")
    assert not baseline[60, 50]
    assert np.all(prepared.mask[baseline])


def test_hysteresis_empty_constant_input_and_padding_exclusion():
    image = np.full((70, 100), 200, np.uint8)
    content = np.zeros_like(image, bool)
    content[10:60, 15:85] = True
    assert not geometry_foreground_mask(image, content, method="otsu_background_hysteresis").any()
    image[20:45, 15:35] = 140
    result = geometry_foreground_mask(image, content, method="otsu_background_hysteresis")
    assert result[30, 15] and not result[~content].any()
    with pytest.raises(ValueError, match="kernel_size"):
        geometry_foreground_mask(image, content, method="otsu_background_hysteresis", background_kernel_size=2)


@pytest.mark.parametrize("denoise,contrast", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("sigma_color", [15, 25])
def test_quality_preprocessing_order_padding_and_configured_path(denoise, contrast, sigma_color):
    rng = np.random.default_rng(42)
    roi = np.clip(rng.normal(180, 4, (60, 90)), 0, 255).astype(np.uint8)
    cv2.ellipse(roi, (45, 30), (20, 12), 0, 0, 360, 140, -1)
    expected_roi = cv2.bilateralFilter(roi, 5, sigma_color, 3) if denoise else roi
    denoised = expected_roi.copy()
    if contrast:
        expected_roi = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4)).apply(expected_roi)
    expected = geometry_foreground_mask(expected_roi, np.ones_like(roi, bool), method="otsu_background")
    method = "otsu_background_clahe" if contrast else "otsu_background"
    settings = validate_preprocessing({"resize": {"enabled": False},
        "filter": {"enabled": denoise, "bilateral": {"params": [5, sigma_color, 3]}},
        "foreground": {"enabled": True, "method": method}}, "geom")
    np.testing.assert_array_equal(prepare_feature_input(roi, settings, "geom").mask, expected)
    content = np.zeros((80, 110), bool)
    content[10:70, 10:100] = True
    original = roi.copy()
    for padding in (0, 255):
        image = np.full(content.shape, padding, np.uint8)
        image[content] = roi.ravel()
        stages = {}
        actual = geometry_foreground_mask(image, content, method=method,
            bilateral_params=[5, sigma_color, 3] if denoise else None, intermediates=stages)
        np.testing.assert_array_equal(actual[content].reshape(roi.shape), expected)
        assert not actual[~content].any()
        assert all(not stage[~content].any() for stage in stages.values())
        if denoise:
            np.testing.assert_array_equal(stages["denoised"][content].reshape(roi.shape), denoised)
        if contrast:
            np.testing.assert_array_equal(stages["contrast"][content].reshape(roi.shape), expected_roi)
    np.testing.assert_array_equal(roi, original)


def test_clahe_constant_content_is_empty_and_bad_kernel_rejected():
    image = np.full((60, 90), 150, np.uint8)
    content = np.ones_like(image, bool)
    assert not geometry_foreground_mask(image, content, method="otsu_background_clahe").any()
    with pytest.raises(ValueError, match="kernel_size"):
        validate_preprocessing({"foreground": {"enabled": True, "method": "otsu_background_clahe",
            "background": {"kernel_size": 2}}}, "geom")


@pytest.mark.parametrize("method,clip,grid", [("otsu_background_clahe_clip1", 1.0, (4, 4)),
                                           ("otsu_background_clahe_grid2", 1.5, (2, 2))])
def test_selected_clahe_settings_and_padding(method, clip, grid):
    roi = np.random.default_rng(42).integers(100, 200, (60, 90), dtype=np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(roi)
    expected = geometry_foreground_mask(enhanced, np.ones_like(roi, bool), method="otsu_background")
    settings = validate_preprocessing({"resize": {"enabled": False},
        "foreground": {"enabled": True, "method": method}}, "geom")
    actual = prepare_feature_input(roi, settings, "geom")
    np.testing.assert_array_equal(actual.intermediates["contrast"], enhanced)
    np.testing.assert_array_equal(actual.mask, expected)
    content = np.zeros((80, 110), bool)
    content[10:70, 10:100] = True
    for padding in (0, 255):
        image = np.full(content.shape, padding, np.uint8)
        image[content] = roi.ravel()
        stages = {}
        mask = geometry_foreground_mask(image, content, method=method, intermediates=stages)
        np.testing.assert_array_equal(mask[content].reshape(roi.shape), expected)
        assert not mask[~content].any()
        assert all(not values[~content].any() for values in stages.values())
