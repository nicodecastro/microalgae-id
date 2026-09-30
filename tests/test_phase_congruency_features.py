"""Numerical regression and dataset I/O tests for the PC descriptors."""

import cv2
import numpy as np
import pandas as pd
import pytest
from phasepack import phasecong

from microalgae.imaging.features.phase_congruency_features import (
    extract_features,
    pc_feature_names,
    phase_congruency_features,
    phase_congruency_moments,
)


@pytest.fixture
def image():
    result = np.full((65, 67), 180, dtype=np.uint8)
    cv2.ellipse(result, (33, 32), (20, 12), 30, 0, 360, 40, -1)
    cv2.line(result, (20, 25), (43, 40), 220, 2)
    return result


def test_matches_phasepack_and_supports_parameter_overrides(image):
    parameters = dict(nscale=4, norient=8, minWaveLength=4, k=2.5)
    expected_M, expected_m, *_ = phasecong(image, **parameters)
    actual_M, actual_m = phase_congruency_moments(image, **parameters)
    np.testing.assert_array_equal(actual_M, expected_M)
    np.testing.assert_array_equal(actual_m, expected_m)
    expected = [expected_M.mean(), expected_M.std(), expected_m.mean(), expected_m.std()]
    np.testing.assert_allclose(phase_congruency_features(image, **parameters), expected)


@pytest.mark.parametrize("value", [0, 127, 255])
def test_constant_image_has_no_phase_features(value):
    np.testing.assert_array_equal(
        phase_congruency_features(np.full((32, 32), value, dtype=np.uint8)),
        np.zeros(4),
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_zero_energy_orientation_matches_covariance_eigenvalues(transpose):
    image = np.zeros((64, 64), dtype=np.uint8)
    image[:, 32:] = 255
    if transpose:
        image = image.T
    with np.errstate(divide="ignore", invalid="ignore"):
        _, _, _, _, pc, *_ = phasecong(image)
    assert any(np.isnan(values).any() for values in pc)
    covariance = np.zeros((*image.shape, 2, 2))
    for i, values in enumerate(pc):
        direction = np.array([np.cos(i * np.pi / 6), np.sin(i * np.pi / 6)])
        # Zero energy contributes zero to the orientation covariance.
        covariance += np.nan_to_num(values)[..., None, None] ** 2 * np.outer(direction, direction)
    eigenvalues = np.linalg.eigvalsh(covariance / 3)
    M, m = phase_congruency_moments(image)
    np.testing.assert_allclose(M, eigenvalues[..., 1] + 5e-5, atol=1e-14)
    np.testing.assert_allclose(m, eigenvalues[..., 0] - 5e-5, atol=1e-14)


@pytest.mark.parametrize("mask_scale", [1, 255])
def test_mask_excludes_pixels_without_altering_input(image, mask_scale):
    original = image.copy()
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[20:40, 22:42] = mask_scale
    M, m = phase_congruency_moments(image)
    roi = mask != 0
    expected = [M[roi].mean(), M[roi].std(ddof=1), m[roi].mean(), m[roi].std(ddof=1)]
    np.testing.assert_allclose(phase_congruency_features(image, mask, ddof=1), expected)
    np.testing.assert_array_equal(image, original)
    assert not np.allclose(expected, phase_congruency_features(image))


@pytest.mark.parametrize("image", [np.zeros((1, 8)), np.zeros((4, 4, 3)), np.full((8, 8), np.nan)])
def test_rejects_invalid_image(image):
    with pytest.raises(ValueError):
        phase_congruency_features(image)


@pytest.mark.parametrize("kwargs", [dict(nscale=1), dict(norient=0), dict(nscale=2.5), dict(ddof=-1)])
def test_rejects_invalid_parameters(image, kwargs):
    with pytest.raises(ValueError):
        phase_congruency_features(image, **kwargs)


def test_rejects_empty_and_mismatched_masks(image):
    for mask in (np.zeros_like(image), np.ones((3, 3))):
        with pytest.raises(ValueError):
            phase_congruency_features(image, mask)


def test_dataset_export_and_previews(tmp_path, image):
    dataset = tmp_path / "raw"
    for label in ("alga_b", "alga_a"):
        (dataset / label).mkdir(parents=True)
        assert cv2.imwrite(str(dataset / label / "b.png"), image)
        assert cv2.imwrite(str(dataset / label / "a.JPG"), np.full_like(image, 100))
        (dataset / label / "ignored.txt").touch()
    output = tmp_path / "features"
    frame = extract_features(dataset, ["alga_b", "alga_a"], output, True, resize=None)
    assert list(frame.columns) == [*pc_feature_names, "Class"]
    assert frame["Class"].tolist() == ["alga_b", "alga_b", "alga_a", "alga_a"]
    np.testing.assert_array_equal(frame.iloc[0, :4].to_numpy(dtype=float), np.zeros(4))
    np.testing.assert_allclose(frame.iloc[1, :4].to_numpy(dtype=float), phase_congruency_features(image))
    pd.testing.assert_frame_equal(frame, pd.read_csv(output / "Phase_congruency_features.csv"))
    assert len(list((output / "phase_congruency").rglob("*.png"))) == 8


def test_dataset_default_resize(tmp_path, image):
    from microalgae.imaging.preprocessing import resize_with_padding

    (tmp_path / "raw" / "alga").mkdir(parents=True)
    assert cv2.imwrite(str(tmp_path / "raw" / "alga" / "image.png"), image)
    frame = extract_features(tmp_path / "raw", ["alga"], tmp_path / "output")
    expected = phase_congruency_features(resize_with_padding(image, (256, 256)))
    np.testing.assert_allclose(frame.iloc[0, :4].to_numpy(dtype=float), expected)


def test_unreadable_image_fails_without_export(tmp_path):
    (tmp_path / "alga").mkdir()
    (tmp_path / "alga" / "corrupt.png").touch()
    with pytest.raises(ValueError, match="Failed to load image"):
        extract_features(tmp_path, ["alga"], tmp_path / "output", resize=None)
    assert not (tmp_path / "output" / "Phase_congruency_features.csv").exists()
