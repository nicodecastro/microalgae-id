"""Chong-style LBP descriptor and export regression checks."""

import cv2
import numpy as np
import pandas as pd
import pytest
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler

from microalgae.imaging.features import lbp_features as lbp


def reference_histogram(image, method="uniform", radius=3, n_points=8):
    """Descriptor copied from the supplied Chong reference, without its script."""
    pattern = local_binary_pattern(image, n_points, radius, method=method)
    hist, _ = np.histogram(
        pattern.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    return hist


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((24, 24), dtype=np.uint8),
        np.arange(576, dtype=np.uint16).reshape(24, 24).astype(np.uint8),
    ],
)
def test_histogram_matches_supplied_reference(image):
    expected = reference_histogram(image)
    np.testing.assert_array_equal(lbp.extract_lbp_histogram(image), expected)
    np.testing.assert_array_equal(
        lbp.extract_lbp_features(image, "sample.png", "class"), expected
    )
    assert len(expected) == 10


def test_public_wrapper_keeps_custom_lbp_parameters():
    image = np.arange(576, dtype=np.uint16).reshape(24, 24).astype(np.uint8)
    expected = reference_histogram(image, method="ror", radius=2, n_points=8)
    np.testing.assert_array_equal(
        lbp.extract_lbp_features(
            image, "sample.png", "class", method="ror", radius=2, n_points=8
        ),
        expected,
    )


def test_export_raw_and_default_scaled_values(tmp_path):
    dataset = tmp_path / "dataset"
    first_class = dataset / "first"
    second_class = dataset / "second"
    first_class.mkdir(parents=True)
    second_class.mkdir()
    images = {
        first_class / "b.png": np.zeros((24, 24), dtype=np.uint8),
        first_class / "a.png": np.arange(576, dtype=np.uint16).reshape(24, 24).astype(np.uint8),
        second_class / "c.png": np.indices((24, 24)).sum(axis=0).astype(np.uint8) * 10,
    }
    for path, image in images.items():
        assert cv2.imwrite(str(path), image)
    (first_class / "notes.txt").write_text("not an image")

    features_dir = tmp_path / "features"
    raw = lbp.extract_features(
        dataset, ["first", "second"], features_dir, standardize=False
    )
    expected_rows = []
    for label in ["first", "second"]:
        class_dir = dataset / label
        for name in lbp.os.listdir(class_dir):
            if name.endswith(".png"):
                image = cv2.imread(str(class_dir / name), cv2.IMREAD_GRAYSCALE)
                expected_rows.append((reference_histogram(cv2.resize(image, (512, 512))), label))
    np.testing.assert_array_equal(
        raw[lbp.FEATURE_NAMES].to_numpy(),
        np.array([hist for hist, _ in expected_rows]),
    )
    assert raw.columns.tolist() == [*lbp.FEATURE_NAMES, "Class"]
    assert raw["Class"].tolist() == [label for _, label in expected_rows]
    pd.testing.assert_frame_equal(
        pd.read_csv(features_dir / "LBP_features.csv"), raw,
        check_exact=False, rtol=1e-14, atol=1e-14,
    )

    scaled = lbp.extract_features(dataset, ["first", "second"], features_dir)
    np.testing.assert_allclose(
        scaled[lbp.FEATURE_NAMES].to_numpy(),
        StandardScaler().fit_transform(raw[lbp.FEATURE_NAMES]),
    )
    assert scaled["Class"].tolist() == raw["Class"].tolist()
    pd.testing.assert_frame_equal(
        pd.read_csv(features_dir / "LBP_features.csv"), scaled,
        check_exact=False, rtol=1e-14, atol=1e-14,
    )


def test_empty_dataset_and_unreadable_image_have_clear_errors(tmp_path):
    class_dir = tmp_path / "dataset" / "class"
    class_dir.mkdir(parents=True)
    (class_dir / "notes.txt").write_text("ignore me")
    with pytest.raises(ValueError, match="No supported images"):
        lbp.extract_features(class_dir.parent, ["class"], tmp_path / "features")

    (class_dir / "broken.png").write_text("not a PNG")
    with pytest.raises(ValueError, match="Failed to load image"):
        lbp.extract_features(class_dir.parent, ["class"], tmp_path / "features")


def test_saved_image_and_failed_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image = np.arange(576, dtype=np.uint16).reshape(24, 24).astype(np.uint8)
    lbp.extract_lbp_features(image, "example.png", "class", save_images=True)
    assert (tmp_path / "data" / "success" / "lbp" / "class" / "example.jpg").is_file()

    monkeypatch.setattr(lbp.cv2, "imwrite", lambda *_args: False)
    with pytest.raises(OSError, match="Failed to save LBP image"):
        lbp.extract_lbp_features(image, "example.png", "class", save_images=True)
