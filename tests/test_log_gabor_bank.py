"""Analytical checks for the fixed-profile MATLAB-toolbox port.

These tests do not imply comparison with an executed MATLAB/Octave runtime.
"""

import cv2
import numpy as np
import pandas as pd
import pytest

from microalgae.imaging.features import gabor_features as gabor
from microalgae.imaging.features._log_gabor_bank import (
    _LogGaborBank, _channel, _log_kernel, _mirror_spectrum, _polar_grid,
)


@pytest.fixture(scope="module")
def bank():
    return _LogGaborBank()


def test_nominal_grid_and_transfer_shape(bank):
    np.testing.assert_array_equal(bank.sf_0, [0.5, 0.25, 0.125, 0.0625])
    np.testing.assert_allclose(np.rad2deg(bank.theta), [
        [0, 15, 0, 15], [30, 45, 30, 45], [60, 75, 60, 75],
        [90, 105, 90, 105], [120, 135, 120, 135], [150, 165, 150, 165],
    ], atol=1e-13)
    assert bank._filters.shape == (256, 256, 6, 4)
    assert np.isfinite(bank._filters).all()
    assert (bank._amplitudes > 0).all()
    np.testing.assert_array_equal(bank._filters[128, 128], np.zeros((6, 4)))
    assert np.any(bank._filters[:, :, :, 0].imag != 0)
    assert not bank._filters[:, :, :, 1:].imag.any()


def test_log_kernel_matches_closed_form_values():
    kernel = _log_kernel(_polar_grid(256), 0, 32)
    assert kernel[128, 160] == pytest.approx(1)
    assert kernel[128, 144] == pytest.approx(np.exp(-0.996 * 0.75))
    assert kernel[96, 128] == pytest.approx(np.exp(-0.996 * 9))
    assert kernel[128, 96] == pytest.approx(np.exp(-0.996 * 36))


@pytest.mark.parametrize("shift", [False, True])
def test_boundary_folding_matches_pointwise_wrapped_gaussians(shift):
    grids = {size: _polar_grid(size) for size in (256, 768)}
    angle = np.pi / 6
    actual = _channel(grids, angle, 128, 1, shift=shift)
    for y, x in [(0, 128), (128, 0), (0, 0), (255, 255), (128, 128)]:
        expected = 0j
        for tile_y in range(3):
            for tile_x in range(3):
                row, column = y + tile_y * 256 - 384, x + tile_x * 256 - 384
                radius = np.hypot(row, column) or 1e-5
                delta = (np.arctan2(-row, column) - angle + np.pi) % (2 * np.pi) - np.pi
                value = np.exp(-0.996 * (0.75 * np.log2(radius / 128) ** 2 + (delta / (np.pi / 6)) ** 2))
                if shift:
                    direction = np.array([-np.sin(angle), np.cos(angle)])
                    value *= np.exp(1j * np.pi * (row * direction[0] + column * direction[1]) * max(abs(direction)) / 256)
                ay = (1 - y / 256, 0, y / 256)[tile_y]
                ax = (1 - x / 256, 0, x / 256)[tile_x]
                weight = (1 + np.cos(min(3 * np.pi * np.hypot(ay, ax), np.pi))) / 2
                expected += value * weight
        assert actual[y, x] == pytest.approx(expected, abs=1e-14)


def test_spectral_mirror_matches_spatial_conjugation():
    spectrum = np.random.default_rng(42).normal(size=(16, 16)) + 1j
    spatial = np.fft.ifft2(np.fft.ifftshift(spectrum))
    expected = np.fft.fftshift(np.fft.fft2(np.conj(spatial)))
    np.testing.assert_allclose(_mirror_spectrum(spectrum), expected, atol=1e-14)
    expected[0, :] = 0
    expected[:, 0] = 0
    np.testing.assert_allclose(_mirror_spectrum(spectrum, include_edges=False), expected, atol=1e-14)


def test_residual_inclusive_normalization_and_calibration(bank):
    power = bank._lowpass ** 2
    for scale in range(4):
        for orientation in range(6):
            transfer = bank._filters[:, :, orientation, scale]
            power = power + np.abs(transfer) ** 2 + np.abs(_mirror_spectrum(transfer)) ** 2
            # A unit coefficient reconstructs twice the real conjugate kernel.
            # Integer translations do not change its peak-to-peak amplitude.
            kernel = np.fft.ifft2(np.fft.ifftshift(np.conj(transfer)))
            assert bank._amplitudes[orientation, scale] == pytest.approx(2 * np.ptp(kernel.real), rel=1e-13)
    np.testing.assert_allclose(power, np.ones((256, 256)), atol=1e-14)


def test_impulse_responses_match_direct_fourier_sums(bank):
    image = np.zeros((256, 256))
    image[0, 0] = 1
    actual = bank.linear_pyramid(image)
    frequency = (np.arange(256) - 128) / 256
    for y, x in [(0, 0), (1, 3), (128, 128), (255, 17)]:
        phase = np.exp(2j * np.pi * (frequency[:, None] * y + frequency[None, :] * x))
        expected = np.einsum("ij,ijts->ts", phase, bank._filters) * bank._amplitudes / 256 ** 2
        np.testing.assert_allclose(actual[y, x], expected, atol=1e-14)


def test_sinusoid_responses_match_two_spectral_lines(bank):
    y, x = np.indices((256, 256))
    phase = 2 * np.pi * (12 * y + 7 * x) / 256
    actual = bank.linear_pyramid(np.cos(phase))
    positive = bank._filters[128 + 12, 128 + 7]
    negative = bank._filters[128 - 12, 128 - 7]
    for row, column in [(0, 0), (5, 9), (128, 128), (255, 17)]:
        expected = 0.5 * (
            positive * np.exp(1j * phase[row, column])
            + negative * np.exp(-1j * phase[row, column])
        ) * bank._amplitudes
        np.testing.assert_allclose(actual[row, column], expected, atol=1e-13)


def test_transform_uses_one_forward_fft(bank, monkeypatch):
    original = np.fft.fft2
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(np.fft, "fft2", count)
    np.testing.assert_array_equal(bank.linear_pyramid(np.full((256, 256), 127)), 0)
    assert len(calls) == 1


@pytest.mark.parametrize("image", [
    np.zeros((128, 256)), np.zeros((256, 256, 3)), np.zeros((0, 0)),
    np.full((256, 256), np.nan), np.zeros((256, 256), dtype=complex),
])
def test_bank_rejects_invalid_input(bank, image):
    with pytest.raises(ValueError):
        bank.linear_pyramid(image)


def test_calculation_helper_uses_native_bank_by_default():
    features = gabor.extract_gabor_statistics(np.zeros((256, 256)))
    assert features.shape == (964,)
    assert np.isfinite(features).all()
    for scale in range(4):
        np.testing.assert_array_equal(features[scale * 241:scale * 241 + 13], np.zeros(13))
        assert features[scale * 241 + 13] == 1


def test_nonfinite_fft_output_is_not_silently_repaired(bank, monkeypatch):
    monkeypatch.setattr(np.fft, "fft2", lambda image: np.full((256, 256), np.nan, dtype=complex))
    with pytest.raises(FloatingPointError):
        bank.linear_pyramid(np.ones((256, 256)))


def test_real_bank_statistics_apply_mask_after_filtering(bank):
    image = np.full((256, 256), 180, dtype=np.uint8)
    cv2.ellipse(image, (128, 128), (60, 35), 30, 0, 360, 40, -1)
    original = image.copy()
    mask = np.zeros_like(image)
    mask[90:160, 90:160] = 255
    pooled = np.sum(np.abs(bank.linear_pyramid(image)), axis=2)
    actual = gabor.extract_gabor_statistics(image, mask, lg=bank)
    assert actual.shape == (964,)
    assert np.isfinite(actual).all()
    for scale in range(4):
        assert actual[scale * 241] == pytest.approx(pooled[:, :, scale][mask != 0].mean())
        assert actual[scale * 241 + 4] == pytest.approx(pooled[:, :, scale][mask != 0].var())
    np.testing.assert_array_equal(image, original)


def test_real_default_dataset_path_reuses_bank_and_exports_raw(tmp_path, monkeypatch, bank):
    monkeypatch.chdir(tmp_path)
    built = []

    def factory():
        built.append(1)
        return bank

    def segmentation(image, dilation_size):
        assert dilation_size == 4
        mask = np.zeros(image.shape, dtype=bool)
        mask[64:192, 64:192] = True
        return mask, image, mask

    monkeypatch.setattr(gabor, "_LogGaborBank", factory)
    monkeypatch.setattr(gabor, "segment_image", segmentation)
    images = []
    for label in ["b", "a"]:
        (tmp_path / "raw" / label).mkdir(parents=True)
        image = np.arange(80 * 120, dtype=np.uint8).reshape(80, 120)
        assert cv2.imwrite(str(tmp_path / "raw" / label / "image.png"), image)
        images.append(image)
    frame = gabor.extract_features(tmp_path / "raw", ["b", "a"], tmp_path / "output", True)
    assert built == [1]
    assert frame.shape == (2, 965)
    assert list(frame.columns) == [*gabor.FEATURE_NAMES, "Class"]
    assert frame.Class.tolist() == ["b", "a"]
    processed = gabor.resize_with_padding(images[0], (256, 256))
    mask, *_ = segmentation(processed, 4)
    expected = gabor.extract_gabor_statistics(processed * mask, mask, lg=bank)
    np.testing.assert_allclose(frame.iloc[0, :-1].to_numpy(dtype=float), expected)
    pd.testing.assert_frame_equal(frame, pd.read_csv(tmp_path / "output" / "Gabor_features.csv"))
    assert len(list((tmp_path / "data" / "success" / "gabor").rglob("*.jpg"))) == 2
    assert not (tmp_path / "data" / "success" / "efd").exists()
