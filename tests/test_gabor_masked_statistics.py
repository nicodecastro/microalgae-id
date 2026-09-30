"""Regression tests for ROI statistics and spatially masked GLCMs."""

import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from skimage.feature import graycomatrix

from features import gabor_features as gabor


def test_first_order_excludes_background_but_retains_foreground_zero():
    band = np.array([[0.0, 2.0], [1000.0, -1000.0]])
    mask = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    stats = gabor.first_order_features(band, bins=2, mask=mask)
    np.testing.assert_allclose(stats[:7], [1, 0, 0, 2, 1, 2, 1])
    np.testing.assert_allclose(stats[7:11], [0.5, 1, 1.5, 1])
    band[mask == 0] = np.nan
    np.testing.assert_array_equal(gabor.first_order_features(band, bins=2, mask=mask), stats)


def test_constant_foreground_has_finite_first_order_statistics():
    band = np.array([[7.0, 7.0], [0.0, 0.0]])
    stats = gabor.first_order_features(band, mask=np.array([[1, 1], [0, 0]]))
    np.testing.assert_allclose(stats, [7, 7, 7, 7, 0, 0, 0, 7, 7, 7, 0, 0, 0])


def test_full_mask_glcms_match_skimage_for_every_offset():
    # Already quantized: endpoints 0 and 15 keep quantization unchanged.
    band = (np.arange(99).reshape(9, 11) % 16).astype(np.uint8)
    expected = graycomatrix(
        band, [1, 3, 5], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16, symmetric=True, normed=True,
    )
    actual = gabor.compute_glcm_matrices(band, mask=np.ones_like(band, dtype=bool))
    np.testing.assert_allclose(actual, expected)


def test_masked_glcms_match_explicit_spatial_pair_counts():
    # Quantization inside ROI produces levels 0 and 3; arbitrary background
    # values must influence neither that range nor the pair frequencies.
    band = 10.0 + 10.0 * (np.indices((7, 8)).sum(axis=0) % 2)
    mask = np.ones(band.shape, dtype=bool)
    mask[1:6, 2] = False
    mask[3, 3:6] = False
    band[~mask] = 1e6
    glcms = gabor.compute_glcm_matrices(band, levels=4, mask=mask)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for d, distance in enumerate([1, 3, 5]):
        for a, angle in enumerate(angles):
            dy, dx = round(distance * np.sin(angle)), round(distance * np.cos(angle))
            expected = np.zeros((4, 4))
            for y, x in np.ndindex(band.shape):
                yn, xn = y + dy, x + dx
                if 0 <= yn < 7 and 0 <= xn < 8 and mask[y, x] and mask[yn, xn]:
                    i, j = int((band[y, x] - 10) * 0.3), int((band[yn, xn] - 10) * 0.3)
                    expected[i, j] += 1
                    expected[j, i] += 1
            expected /= expected.sum()
            np.testing.assert_allclose(glcms[:, :, d, a], expected)
    band[~mask] = np.nan
    np.testing.assert_array_equal(gabor.compute_glcm_matrices(band, levels=4, mask=mask), glcms)


def test_valid_zero_pairs_are_not_removed():
    band = np.array([[0.0, 0.0, 1.0]])
    glcm = gabor.compute_glcm_matrices(band, levels=2, mask=np.ones_like(band))[:, :, 0, 0]
    np.testing.assert_allclose(glcm, [[0.5, 0.25], [0.25, 0]])


def test_constant_roi_maps_to_zero_level_and_normalizes_valid_pairs():
    band = np.full((8, 8), 42.0)
    glcms = gabor.compute_glcm_matrices(band)
    np.testing.assert_array_equal(glcms[0, 0], np.ones((3, 4)))
    np.testing.assert_allclose(glcms.sum(axis=(0, 1)), 1)
    assert np.isfinite(gabor.second_order_features(band)).all()


def test_single_pixel_and_no_pair_offsets_have_explicit_zero_features():
    band = np.array([[5.0]])
    assert np.isfinite(gabor.first_order_features(band)).all()
    assert not gabor.compute_glcm_matrices(band).any()
    np.testing.assert_array_equal(gabor.second_order_features(band), np.zeros(228))
    # A one-row image supports horizontal neighbors only, without wrapping.
    glcms = gabor.compute_glcm_matrices(np.array([[0.0, 1.0]]), levels=2)
    assert glcms[:, :, 0, 0].sum() == 1
    assert not glcms[:, :, 1:, :].any()
    assert not glcms[:, :, :, 1:].any()


@pytest.mark.parametrize("function", [gabor.first_order_features, gabor.compute_glcm_matrices, gabor.second_order_features])
def test_empty_or_invalid_mask_is_rejected(function):
    band = np.ones((4, 4))
    for mask in (np.zeros((4, 4)), np.ones((3, 3)), np.full((4, 4), np.nan)):
        with pytest.raises(ValueError):
            function(band, mask=mask)


class FakeLogGabor:
    """Known orientation responses let us test pooling/statistics independently."""

    def __init__(self, *args):
        pass

    def linear_pyramid(self, image):
        return np.broadcast_to(
            image[:, :, None, None] * np.arange(1, 5)[None, None, None, :] / 6,
            (*image.shape, 6, 4),
        ).astype(complex)


def test_each_pooled_band_receives_the_mask():
    image = np.arange(64, dtype=float).reshape(8, 8)
    mask = np.zeros(image.shape, dtype=bool)
    mask[2:6, 2:6] = True
    actual = gabor.extract_gabor_features(image, mask, "test.png", "alga", False, FakeLogGabor())
    assert actual.shape == (964,)
    assert np.isfinite(actual).all()
    for scale in range(4):
        assert actual[scale * 241] == pytest.approx(image[mask].mean() * (scale + 1))
    changed = image.copy()
    changed[~mask] = -10000
    np.testing.assert_allclose(
        actual, gabor.extract_gabor_features(changed, mask, "test.png", "alga", False, FakeLogGabor()),
    )


def test_empty_mask_fails_before_filtering():
    with pytest.raises(ValueError, match="empty foreground"):
        gabor.extract_gabor_features(np.ones((8, 8)), np.zeros((8, 8)), "test.png", "alga", False, None)


def test_export_labels_match_per_scale_feature_order(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "LogGabor", SimpleNamespace(LogGabor=FakeLogGabor))
    def segmentation(image, dilation_size):
        mask = np.zeros(image.shape, dtype=bool)
        mask[100:150, 100:150] = True
        return mask, image, mask
    monkeypatch.setattr(gabor, "segment_image", segmentation)
    (tmp_path / "raw" / "alga").mkdir(parents=True)
    assert cv2.imwrite(str(tmp_path / "raw" / "alga" / "image.png"), np.full((256, 256), 6, dtype=np.uint8))
    gabor.extract_features(tmp_path / "raw", ["alga"], tmp_path / "features")
    frame = pd.read_csv(tmp_path / "features" / "Gabor_features.csv")
    assert frame.shape == (1, 965)
    assert frame.loc[0, "Class"] == "alga"
    for scale in range(1, 5):
        assert frame.columns[(scale - 1) * 241] == f"S{scale}_1st_mean"
        assert frame.loc[0, f"S{scale}_1st_mean"] == 6 * scale
        assert frame.loc[0, f"S{scale}_2nd_energy_d1_a0"] == 1
