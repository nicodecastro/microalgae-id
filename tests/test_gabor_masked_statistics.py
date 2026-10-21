"""Regression tests for ROI statistics and spatially masked GLCMs."""

from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from skimage.feature import graycomatrix

from microalgae.imaging.features import gabor_features as gabor


def test_first_order_excludes_background_but_retains_foreground_zero():
    band = np.array([[0.0, 2.0], [1000.0, -1000.0]])
    mask = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    stats = gabor.first_order_features(band, bins=2, mask=mask)
    np.testing.assert_allclose(stats[:7], [1, 0.5, 0, 2, 1, 2, 1])
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
    monkeypatch.setattr(gabor, "_LogGaborBank", FakeLogGabor)
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


def test_histogram_mode_and_pearson_kurtosis():
    values = np.array([[0, 10.01, 10.02, 10.03, 10.04]])
    stats = gabor.first_order_features(values)
    assert stats[1] == pytest.approx(10.04 - (10.04 / 255) / 2)
    assert stats[6] == pytest.approx(-0.2 * np.log2(0.2) - 0.8 * np.log2(0.8))
    centered = values - values.mean()
    assert stats[12] == pytest.approx(np.mean(centered ** 4) / np.var(values) ** 2)


@pytest.mark.parametrize("bins", [True, 0, 1, 2.5, np.nan])
def test_rejects_invalid_histogram_bins(bins):
    with pytest.raises(ValueError, match="bins"):
        gabor.first_order_features(np.ones((8, 8)), bins=bins)


@pytest.mark.parametrize("function", [gabor.first_order_features, gabor.compute_glcm_matrices, gabor.second_order_features])
@pytest.mark.parametrize("band", [np.ones((8, 8), dtype=complex), np.full((8, 8), "1"), np.ones((8, 8, 3))])
def test_rejects_non_real_or_non_image_bands(function, band):
    with pytest.raises(ValueError, match="2-D real numeric"):
        function(band)


@pytest.mark.parametrize("responses", [
    np.zeros((8, 8, 6, 3)), np.zeros((8, 8, 4, 6)), np.zeros((8, 8)),
    np.full((8, 8, 6, 4), np.nan), np.zeros((7, 8, 6, 4)),
])
def test_rejects_malformed_or_nonfinite_bank_responses(responses):
    supplied = SimpleNamespace(linear_pyramid=lambda image: responses)
    with pytest.raises(ValueError, match="response"):
        gabor.extract_gabor_statistics(np.ones((8, 8)), lg=supplied)


def test_rejects_invalid_bank_and_nonfinite_filter_input():
    with pytest.raises(ValueError, match="linear_pyramid"):
        gabor.extract_gabor_statistics(np.ones((8, 8)), lg={})
    image = np.ones((8, 8))
    image[0, 0] = np.nan
    mask = np.ones_like(image)
    mask[0, 0] = 0
    with pytest.raises(ValueError, match="finite"):
        gabor.extract_gabor_statistics(image, mask, lg=FakeLogGabor())


def test_complex_orientation_pooling_sums_magnitudes_not_complex_values():
    responses = np.zeros((2, 2, 6, 4), dtype=complex)
    responses[:, :, 0] = 3 + 4j
    responses[:, :, 1] = -3 - 4j
    np.testing.assert_array_equal(gabor.collapse_orientations(responses), np.full((2, 2, 4), 10))


def test_dataset_preserves_class_directory_order_and_binary_mask_encodings(tmp_path, monkeypatch):
    monkeypatch.setattr(gabor, "_LogGaborBank", FakeLogGabor)
    monkeypatch.setattr(
        gabor, "segment_image",
        lambda image, dilation_size: (np.full(image.shape, 255, dtype=np.uint8), image, image),
    )
    for label in ["z", "a"]:
        path = tmp_path / "raw" / label
        path.mkdir(parents=True)
        assert cv2.imwrite(str(path / "b.png"), np.full((16, 24), 9, dtype=np.uint8))
        assert cv2.imwrite(str(path / "a.PNG"), np.full((256, 256), 2, dtype=np.uint8))
        (path / "ignored.txt").touch()
        (path / "directory.png").mkdir()
    monkeypatch.setattr(gabor.os, "listdir", lambda path: ["b.png", "ignored.txt", "directory.png", "a.PNG"])
    frame = gabor.extract_features(tmp_path / "raw", ["z", "a"], tmp_path / "output")
    assert frame.Class.tolist() == ["z", "z", "a", "a"]
    # Resizing the rectangular image pads one third of its 256 x 256 canvas.
    np.testing.assert_allclose(frame.S1_1st_mean, [9 * 170 / 256, 2, 9 * 170 / 256, 2])
    assert frame.columns[0] == "S1_1st_mean"
    assert frame.columns[241] == "S2_1st_mean"
    assert frame.columns[-1] == "Class"


@pytest.mark.parametrize("class_subfolders", [[], ["alga"]])
def test_empty_dataset_does_not_export(tmp_path, class_subfolders):
    (tmp_path / "alga").mkdir()
    (tmp_path / "alga" / "not_image.txt").touch()
    with pytest.raises(ValueError, match="No supported images"):
        gabor.extract_features(tmp_path, class_subfolders, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_unreadable_image_reports_path_without_export(tmp_path):
    (tmp_path / "alga").mkdir()
    path = tmp_path / "alga" / "corrupt.png"
    path.touch()
    with pytest.raises(ValueError) as error:
        gabor.extract_features(tmp_path, ["alga"], tmp_path / "output")
    assert "Failed to load image" in str(error.value)
    assert str(path) in str(error.value)
    assert not (tmp_path / "output").exists()


def test_preview_uint8_and_correct_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saved = []
    monkeypatch.setattr(cv2, "imwrite", lambda path, image: saved.append((path, image.copy())) or True)
    image = np.array([[-5.2, 300], [10.4, 10.6]])
    gabor.extract_gabor_features(image, None, "input.png", "alga", True, FakeLogGabor())
    path, preview = saved[0]
    assert Path(path) == Path("data/success/gabor/alga/input.jpg")
    assert preview.dtype == np.uint8
    np.testing.assert_array_equal(preview, [[0, 255], [10, 11]])


def test_failed_preview_reports_source_image_without_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "alga").mkdir()
    path = tmp_path / "alga" / "input.png"
    assert cv2.imwrite(str(path), np.ones((256, 256), dtype=np.uint8))
    monkeypatch.setattr(gabor, "_LogGaborBank", FakeLogGabor)
    monkeypatch.setattr(gabor, "segment_image", lambda image, dilation_size: (np.ones_like(image), image, image))
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError) as error:
        gabor.extract_features(tmp_path, ["alga"], tmp_path / "output", True)
    assert "Failed to save Log-Gabor preview" in str(error.value)
    assert str(path) in str(error.value)
    assert not (tmp_path / "output").exists()


def test_filter_failure_reports_source_image_without_export(tmp_path, monkeypatch):
    (tmp_path / "alga").mkdir()
    path = tmp_path / "alga" / "input.png"
    assert cv2.imwrite(str(path), np.ones((256, 256), dtype=np.uint8))
    monkeypatch.setattr(gabor, "_LogGaborBank", FakeLogGabor)
    monkeypatch.setattr(gabor, "segment_image", lambda image, dilation_size: (np.ones_like(image), image, image))

    def fail(self, image):
        raise FloatingPointError("unexplained failure")

    monkeypatch.setattr(FakeLogGabor, "linear_pyramid", fail)
    with pytest.raises(FloatingPointError) as error:
        gabor.extract_features(tmp_path, ["alga"], tmp_path / "output")
    assert "unexplained failure" in str(error.value)
    assert str(path) in str(error.value)
    assert not (tmp_path / "output").exists()
