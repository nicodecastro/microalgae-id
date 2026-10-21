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


def test_default_matches_explicit_four_scale_phasepack(image):
    expected_M, expected_m, *_ = phasecong(
        image, nscale=4, norient=6, minWaveLength=3, mult=2.1,
        sigmaOnf=0.55, k=2, cutOff=0.5, g=10, noiseMethod=-1,
    )
    actual_M, actual_m = phase_congruency_moments(image)
    np.testing.assert_array_equal(actual_M, expected_M)
    np.testing.assert_array_equal(actual_m, expected_m)
    expected = [expected_M.mean(), expected_M.std(), expected_m.mean(), expected_m.std()]
    np.testing.assert_allclose(phase_congruency_features(image), expected)


def test_five_scale_override_preserves_previous_phasepack_default(image):
    expected_M, expected_m, *_ = phasecong(image)
    actual_M, actual_m = phase_congruency_moments(image, nscale=5)
    np.testing.assert_array_equal(actual_M, expected_M)
    np.testing.assert_array_equal(actual_m, expected_m)
    default_M, _ = phase_congruency_moments(image)
    assert not np.allclose(default_M, actual_M)


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
        _, _, _, _, pc, *_ = phasecong(image, nscale=4, norient=6)
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


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("kwargs", [
    dict(nscale=1), dict(norient=0), dict(nscale=2.5), dict(norient=True),
    dict(minWaveLength=1.9), dict(mult=1), dict(sigmaOnf=0), dict(sigmaOnf=1),
    dict(k=-1), dict(cutOff=-0.1), dict(cutOff=1.1), dict(g=0),
    dict(noiseMethod=-3), dict(noiseMethod=-1.5), dict(unknown_parameter=1),
])
def test_rejects_invalid_filter_parameters(image, constant, kwargs):
    with pytest.raises(ValueError):
        phase_congruency_features(np.zeros_like(image) if constant else image, **kwargs)


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("name", ["minWaveLength", "mult", "sigmaOnf", "k", "cutOff", "g", "noiseMethod"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_filter_parameters_must_be_finite(image, constant, name, value):
    with pytest.raises(ValueError, match=name):
        phase_congruency_features(np.zeros_like(image) if constant else image, **{name: value})


@pytest.mark.parametrize("value", [True, "2", [2], np.array(2), 2 + 0j])
def test_filter_parameters_must_be_real_scalars(image, value):
    with pytest.raises(ValueError, match="k"):
        phase_congruency_features(image, k=value)


@pytest.mark.parametrize("noise_method", [-1, -2, 0, 0.2])
def test_noise_methods_and_valid_parameter_boundaries(image, noise_method):
    parameters = dict(
        nscale=np.int64(4), minWaveLength=2, mult=2, sigmaOnf=0.5,
        k=0, cutOff=0, g=1, noiseMethod=noise_method,
    )
    expected_M, expected_m, *_ = phasecong(image, **parameters)
    actual_M, actual_m = phase_congruency_moments(image, **parameters)
    np.testing.assert_array_equal(actual_M, expected_M)
    np.testing.assert_array_equal(actual_m, expected_m)


@pytest.mark.parametrize("ddof", [-1, True, 0.5, 65 * 67])
def test_rejects_invalid_ddof(image, ddof):
    with pytest.raises(ValueError, match="ddof"):
        phase_congruency_features(image, ddof=ddof)


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
    assert list(frame.columns) == [*pc_feature_names, "image_id", "Class"]
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


@pytest.mark.parametrize("parameters", [
    [], False, 0, {"nscale": 1}, {"unknown_parameter": 1},
    {"minWaveLength": np.nan}, {1: "not a parameter name"},
])
def test_dataset_validates_parameters_before_reading_files(tmp_path, parameters):
    with pytest.raises(ValueError):
        extract_features(
            tmp_path / "missing", ["alga"], tmp_path / "output",
            phasecong_parameters=parameters,
        )
    assert not (tmp_path / "output").exists()


def test_dataset_supports_five_scale_parameter_mapping(tmp_path, image):
    (tmp_path / "raw" / "alga").mkdir(parents=True)
    assert cv2.imwrite(str(tmp_path / "raw" / "alga" / "image.png"), image)
    parameters = {"nscale": 5, "noiseMethod": 0}
    frame = extract_features(
        tmp_path / "raw", ["alga"], tmp_path / "output", resize=None,
        phasecong_parameters=parameters,
    )
    expected = phase_congruency_features(image, **parameters)
    np.testing.assert_allclose(frame.iloc[0, :4].to_numpy(dtype=float), expected)
    assert parameters == {"nscale": 5, "noiseMethod": 0}


def test_failed_preview_write_fails_without_export(tmp_path, image, monkeypatch):
    (tmp_path / "alga").mkdir()
    assert cv2.imwrite(str(tmp_path / "alga" / "image.png"), image)
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError, match="image.png_maximum.png"):
        extract_features(tmp_path, ["alga"], tmp_path / "output", True, resize=None)
    assert not (tmp_path / "output" / "Phase_congruency_features.csv").exists()


@pytest.mark.parametrize("invalid_pc", [False, True])
def test_rejects_unexplained_numerical_failures(image, monkeypatch, invalid_pc):
    import phasepack

    invalid_moment = np.full(image.shape, np.nan)
    pc = [np.zeros(image.shape) for _ in range(6)]
    if invalid_pc:
        pc[0][0, 0] = np.nan
    responses = [[np.ones(image.shape)] for _ in range(6)]
    monkeypatch.setattr(
        phasepack, "phasecong",
        lambda *args, **kwargs: (invalid_moment, invalid_moment, None, None, pc, responses, None),
    )
    with pytest.raises(FloatingPointError, match="nonzero energy" if invalid_pc else "without zero-amplitude"):
        phase_congruency_moments(image)


def test_dataset_numerical_error_includes_image_path(tmp_path, image, monkeypatch):
    import phasepack

    (tmp_path / "alga").mkdir()
    path = tmp_path / "alga" / "image.png"
    assert cv2.imwrite(str(path), image)

    def fail(*args, **kwargs):
        raise FloatingPointError("unexplained failure")

    monkeypatch.setattr(phasepack, "phasecong", fail)
    with pytest.raises(FloatingPointError) as error:
        extract_features(tmp_path, ["alga"], tmp_path / "output", resize=None)
    assert str(path) in str(error.value)
    assert "unexplained failure" in str(error.value)
    assert not (tmp_path / "output" / "Phase_congruency_features.csv").exists()
