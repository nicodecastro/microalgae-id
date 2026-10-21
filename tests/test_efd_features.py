"""Equation, contour-invariance, and export checks for EFD amplitudes."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from microalgae.imaging.features import efd_features as efd


CONTOUR = np.array([[0., 0.], [6., 0.], [7., 2.], [3., 5.], [0., 3.]])


def paper_amplitudes(contour):
    """Evaluate Sanchez's equations 4-8 directly, including the closing segment."""
    segments = np.diff(np.vstack([contour, contour[0]]), axis=0)
    durations = np.linalg.norm(segments, axis=1)
    times = np.r_[0., np.cumsum(durations)]
    perimeter = times[-1]
    result = []
    for harmonic in range(1, 31):
        factor = perimeter / (2 * harmonic ** 2 * np.pi ** 2)
        phase = 2 * harmonic * np.pi * times / perimeter
        coefficients = [
            factor * sum(
                segments[index, axis] / durations[index]
                * (function(phase[index + 1]) - function(phase[index]))
                for index in range(len(segments))
            )
            for axis in [0, 1]
            for function in [np.cos, np.sin]
        ]
        result.append(0.5 * np.linalg.norm(coefficients))
    return result


def example_mask():
    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.ellipse(mask, (55, 60), (30, 18), 23, 0, 360, 1, -1)
    return mask


def test_raw_amplitudes_match_published_equations():
    actual = efd.extract_efd_amplitudes(CONTOUR)
    assert len(actual) == 30
    np.testing.assert_allclose(actual, paper_amplitudes(CONTOUR), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("normalize_shape", [False, True])
def test_amplitudes_ignore_position_rotation_start_and_traversal(normalize_shape):
    angle = 0.72
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    expected = efd.extract_efd_amplitudes(CONTOUR, normalize_shape=normalize_shape)
    for transformed in [
        CONTOUR + [123.4, -87.6], CONTOUR @ rotation,
        np.roll(CONTOUR, 2, axis=0), CONTOUR[::-1],
    ]:
        np.testing.assert_allclose(
            efd.extract_efd_amplitudes(transformed, normalize_shape=normalize_shape),
            expected, rtol=1e-10, atol=1e-12,
        )


def test_only_normalized_amplitudes_ignore_size():
    raw = np.array(efd.extract_efd_amplitudes(CONTOUR))
    np.testing.assert_allclose(
        efd.extract_efd_amplitudes(CONTOUR * 2.5), raw * 2.5,
        rtol=1e-10, atol=1e-12,
    )
    np.testing.assert_allclose(
        efd.extract_efd_amplitudes(CONTOUR * 2.5, normalize_shape=True),
        efd.extract_efd_amplitudes(CONTOUR, normalize_shape=True),
        rtol=1e-10, atol=1e-12,
    )


def test_duplicate_points_and_explicit_closure_do_not_change_features():
    repeated = np.vstack([CONTOUR[0], CONTOUR, CONTOUR[-1], CONTOUR[0], CONTOUR[0]])
    np.testing.assert_array_equal(
        efd.extract_efd_amplitudes(repeated), efd.extract_efd_amplitudes(CONTOUR)
    )


@pytest.mark.parametrize("contour", [
    np.empty((0, 2)), [1, 2], [[0, 0], [1, 1]],
    [[0, 0], [0, 0], [0, 0]], [[0, 0], [1, 1], [2, 2]],
    [[0, 0], [1, 1], [2, np.nan]],
])
def test_invalid_contours_are_rejected(contour):
    with pytest.raises(ValueError, match="contour must"):
        efd.extract_efd_amplitudes(contour)


@pytest.mark.parametrize("normalize_shape", [False, True])
def test_mask_encodings_largest_region_and_external_outline(normalize_shape):
    mask = example_mask()
    expected = efd.extract_efd_features(
        mask, "sample.png", "class", normalize_shape=normalize_shape
    )
    for encoded in [mask.astype(bool), mask * 255, mask.astype(np.float64)]:
        np.testing.assert_array_equal(
            efd.extract_efd_features(
                encoded, "sample.png", "class", normalize_shape=normalize_shape
            ), expected,
        )
    cv2.circle(mask, (135, 90), 6, 1, -1)
    cv2.circle(mask, (55, 60), 5, 0, -1)
    np.testing.assert_array_equal(
        efd.extract_efd_features(mask, "sample.png", "class", normalize_shape=normalize_shape),
        expected,
    )


@pytest.mark.parametrize("kind", ["empty", "point", "line"])
def test_missing_or_zero_area_outline_has_image_context(kind):
    mask = np.zeros((16, 16), dtype=np.uint8)
    if kind == "point":
        mask[8, 8] = 1
    elif kind == "line":
        mask[8, 2:14] = 1
    with pytest.raises(ValueError, match="sample.png"):
        efd.extract_efd_features(mask, "sample.png", "class")


@pytest.mark.parametrize("mask", [
    np.zeros((2, 2, 3), dtype=np.uint8), np.empty((0, 2)),
    np.array([[0., 0.5], [1., 0.]]), np.array([[0., np.nan], [1., 0.]]),
])
def test_nonbinary_masks_are_rejected(mask):
    with pytest.raises(ValueError, match="binary mask"):
        efd.extract_efd_features(mask, "sample.png", "class")


def test_nonfinite_library_results_are_rejected(monkeypatch):
    monkeypatch.setattr(
        efd, "elliptic_fourier_descriptors", lambda *_args, **_kwargs: np.full((30, 4), np.nan)
    )
    with pytest.raises(ValueError, match="Non-finite EFD"):
        efd.extract_efd_amplitudes(CONTOUR)


def test_exports_keep_raw_values_and_separate_normalized_results(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    paths = [dataset / "second" / "c.png", dataset / "first" / "b.png", dataset / "first" / "a.png"]
    for path, axes in zip(paths, [(20, 10), (15, 7), (23, 5)]):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((64, 64), 255, dtype=np.uint8)
        cv2.ellipse(image, (32, 32), axes, 17, 0, 360, 0, -1)
        assert cv2.imwrite(str(path), image)
    (dataset / "first" / "notes.txt").write_text("ignored")
    original_listdir = efd.os.listdir
    monkeypatch.setattr(
        efd.os, "listdir",
        lambda path: ["b.png", "notes.txt", "a.png"] if Path(path) == dataset / "first"
        else original_listdir(path),
    )
    output_dir = tmp_path / "features"
    raw = efd.extract_features(dataset, ["second", "first"], output_dir)
    raw_path = output_dir / "EFD_features.csv"
    raw_bytes = raw_path.read_bytes()
    normalized = efd.extract_features(
        dataset, ["second", "first"], output_dir, normalize_shape=True
    )
    assert raw_path.read_bytes() == raw_bytes
    assert raw.columns.tolist() == normalized.columns.tolist() == [*efd.FEATURE_NAMES, "image_id", "Class"]
    assert raw["Class"].tolist() == normalized["Class"].tolist() == ["second", "first", "first"]
    for flag, frame, filename in [
        (False, raw, "EFD_features.csv"),
        (True, normalized, "EFD_shape_normalized_features.csv"),
    ]:
        expected = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            mask, _, _ = efd.segment_image(image, dilation_size=4)
            expected.append(efd.extract_efd_features(mask, path.name, path.parent.name, normalize_shape=flag))
        np.testing.assert_array_equal(frame[efd.FEATURE_NAMES].to_numpy(), expected)
        pd.testing.assert_frame_equal(
            pd.read_csv(output_dir / filename), frame,
            check_exact=False, rtol=1e-12, atol=1e-12,
        )


def test_empty_dataset_and_unreadable_image_fail_before_export(tmp_path):
    class_dir = tmp_path / "dataset" / "class"
    class_dir.mkdir(parents=True)
    (class_dir / "notes.txt").write_text("ignored")
    output_dir = tmp_path / "features"
    with pytest.raises(ValueError, match="No supported images"):
        efd.extract_features(class_dir.parent, ["class"], output_dir)
    (class_dir / "broken.png").write_text("not a PNG")
    with pytest.raises(ValueError, match="Failed to load image"):
        efd.extract_features(class_dir.parent, ["class"], output_dir)
    assert not (output_dir / "EFD_features.csv").exists()
    assert (output_dir / "EFD_features_failures.csv").is_file()


def test_preview_uses_uint8_and_reports_failed_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mask = example_mask().astype(bool)
    original_imwrite = efd.cv2.imwrite
    saved_arrays = []

    def save_image(path, image):
        saved_arrays.append(image.copy())
        return original_imwrite(path, image)

    monkeypatch.setattr(efd.cv2, "imwrite", save_image)
    efd.extract_efd_features(mask, "sample.png", "class", True)
    assert (tmp_path / "data" / "success" / "efd" / "class" / "sample.jpg").is_file()
    assert saved_arrays[0].dtype == np.uint8
    np.testing.assert_array_equal(np.unique(saved_arrays[0]), [0, 255])
    monkeypatch.setattr(efd.cv2, "imwrite", lambda *_args: False)
    with pytest.raises(OSError, match="Failed to save EFD mask"):
        efd.extract_efd_features(mask, "sample.png", "class", True)
