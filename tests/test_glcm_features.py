"""Chong-style GLCM descriptor and export regression checks."""

import cv2
import numpy as np
import pandas as pd
import pytest
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

from microalgae.imaging.features import glcm_features as glcm_features


def reference_features(image, properties=glcm_features.PROPERTIES):
    """Feature calculation from the supplied Chong script, without its loop."""
    glcm = graycomatrix(
        image,
        distances=[1, 3, 5],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    return [graycoprops(glcm, prop).ravel()[0] for prop in properties]


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((16, 16), dtype=np.uint8),
        np.arange(256, dtype=np.uint8).reshape(16, 16),
    ],
)
def test_properties_match_chong_reference(image):
    expected = reference_features(image)
    np.testing.assert_array_equal(glcm_features.extract_glcm_statistics(image), expected)
    np.testing.assert_array_equal(
        glcm_features.extract_glcm_features(
            image, "sample.png", "class", glcm_features.PROPERTIES
        ),
        expected,
    )
    assert len(expected) == 5


def test_only_first_distance_and_angle_are_selected():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    matrix = graycomatrix(
        image,
        distances=[1, 3, 5],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast = graycoprops(matrix, "contrast")
    assert glcm_features.extract_glcm_statistics(image, ["contrast"]) == [contrast[0, 0]]
    assert contrast[0, 0] != contrast[0, 2]


def test_export_raw_and_default_scaled_values(tmp_path):
    dataset = tmp_path / "dataset"
    first_class = dataset / "first"
    second_class = dataset / "second"
    first_class.mkdir(parents=True)
    second_class.mkdir()
    images = {
        first_class / "b.png": np.zeros((16, 16), dtype=np.uint8),
        first_class / "a.png": np.arange(256, dtype=np.uint8).reshape(16, 16),
        second_class / "c.png": np.indices((16, 16)).sum(axis=0).astype(np.uint8) * 8,
    }
    for path, image in images.items():
        assert cv2.imwrite(str(path), image)
    (first_class / "notes.txt").write_text("not an image")

    features_dir = tmp_path / "features"
    raw = glcm_features.extract_features(
        dataset, ["first", "second"], features_dir, standardize=False
    )
    expected_rows = []
    for label in ["first", "second"]:
        class_dir = dataset / label
        for name in glcm_features.os.listdir(class_dir):
            if name.endswith(".png"):
                image = cv2.imread(str(class_dir / name), cv2.IMREAD_GRAYSCALE)
                expected_rows.append((reference_features(cv2.resize(image, (512, 512))), label))
    np.testing.assert_array_equal(
        raw[glcm_features.PROPERTIES].to_numpy(),
        np.array([features for features, _ in expected_rows]),
    )
    assert raw.columns.tolist() == [*glcm_features.PROPERTIES, "image_id", "Class"]
    assert raw["Class"].tolist() == [label for _, label in expected_rows]
    pd.testing.assert_frame_equal(
        pd.read_csv(features_dir / "GLCM_features.csv"), raw,
        check_exact=False, rtol=1e-14, atol=1e-14,
    )

    scaled = glcm_features.extract_features(dataset, ["first", "second"], features_dir)
    np.testing.assert_allclose(
        scaled[glcm_features.PROPERTIES].to_numpy(),
        StandardScaler().fit_transform(raw[glcm_features.PROPERTIES]),
    )
    assert scaled["Class"].tolist() == raw["Class"].tolist()
    pd.testing.assert_frame_equal(
        pd.read_csv(features_dir / "GLCM_features.csv"), scaled,
        check_exact=False, rtol=1e-14, atol=1e-14,
    )


def test_empty_dataset_and_unreadable_image_have_clear_errors(tmp_path):
    class_dir = tmp_path / "dataset" / "class"
    class_dir.mkdir(parents=True)
    (class_dir / "notes.txt").write_text("ignore me")
    with pytest.raises(ValueError, match="No supported images"):
        glcm_features.extract_features(class_dir.parent, ["class"], tmp_path / "features")

    (class_dir / "broken.png").write_text("not a PNG")
    with pytest.raises(ValueError, match="Failed to load image"):
        glcm_features.extract_features(class_dir.parent, ["class"], tmp_path / "features")


def test_saved_image_and_failed_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    glcm_features.extract_glcm_features(
        image, "example.png", "class", glcm_features.PROPERTIES, save_images=True
    )
    assert (tmp_path / "data" / "success" / "glcm" / "class" / "example.jpg").is_file()

    monkeypatch.setattr(glcm_features.cv2, "imwrite", lambda *_args: False)
    with pytest.raises(OSError, match="Failed to save GLCM image"):
        glcm_features.extract_glcm_features(
            image, "example.png", "class", glcm_features.PROPERTIES, save_images=True
        )


def test_prepared_image_requires_grayscale_uint8():
    with pytest.raises(ValueError, match="2-D uint8"):
        glcm_features.extract_glcm_statistics(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="2-D uint8"):
        glcm_features.extract_glcm_statistics(np.zeros((8, 8), dtype=np.float32))


def test_orientation_pooled_statistics_average_properties_per_distance():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    matrix = graycomatrix(
        image,
        distances=[1, 3, 5],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    expected = [
        value
        for prop in glcm_features.PROPERTIES
        for value in graycoprops(matrix, prop).mean(axis=1)
    ]
    actual = glcm_features.extract_glcm_statistics(image, mode="orientation_pooled")
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        glcm_features.extract_glcm_features(
            image, "sample.png", "class", glcm_features.PROPERTIES,
            mode="orientation_pooled",
        ),
        expected,
    )
    assert len(actual) == 15
    assert actual[0] != glcm_features.extract_glcm_statistics(image)[0]


def test_both_modes_export_separately_with_same_row_order(tmp_path):
    dataset = tmp_path / "dataset"
    first_class = dataset / "first"
    second_class = dataset / "second"
    first_class.mkdir(parents=True)
    second_class.mkdir()
    images = {
        first_class / "b.png": np.zeros((16, 16), dtype=np.uint8),
        first_class / "a.png": np.arange(256, dtype=np.uint8).reshape(16, 16),
        second_class / "c.png": np.indices((16, 16)).sum(axis=0).astype(np.uint8) * 8,
    }
    for path, image in images.items():
        assert cv2.imwrite(str(path), image)

    features_dir = tmp_path / "features"
    chong = glcm_features.extract_features(
        dataset, ["first", "second"], features_dir, standardize=False
    )
    chong_csv = features_dir / "GLCM_features.csv"
    chong_csv_before = chong_csv.read_bytes()
    pooled_raw = glcm_features.extract_features(
        dataset, ["first", "second"], features_dir,
        standardize=False, mode="orientation_pooled",
    )
    pooled_csv = features_dir / "GLCM_orientation_pooled_features.csv"
    expected_names = [
        f"{prop}_d{distance}"
        for prop in glcm_features.PROPERTIES
        for distance in [1, 3, 5]
    ]
    assert pooled_raw.columns.tolist() == [*expected_names, "image_id", "Class"]
    assert pooled_raw["Class"].tolist() == chong["Class"].tolist()
    assert chong_csv.read_bytes() == chong_csv_before
    pd.testing.assert_frame_equal(
        pd.read_csv(pooled_csv), pooled_raw,
        check_exact=False, rtol=1e-14, atol=1e-14,
    )

    expected_rows = []
    for label in ["first", "second"]:
        class_dir = dataset / label
        for name in glcm_features.os.listdir(class_dir):
            image = cv2.imread(str(class_dir / name), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (512, 512))
            matrix = graycomatrix(
                image, distances=[1, 3, 5],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=256, symmetric=True, normed=True,
            )
            expected_rows.append([
                value
                for prop in glcm_features.PROPERTIES
                for value in graycoprops(matrix, prop).mean(axis=1)
            ])
    np.testing.assert_array_equal(pooled_raw[expected_names].to_numpy(), expected_rows)

    pooled_scaled = glcm_features.extract_features(
        dataset, ["first", "second"], features_dir, mode="orientation_pooled"
    )
    np.testing.assert_allclose(
        pooled_scaled[expected_names].to_numpy(),
        StandardScaler().fit_transform(pooled_raw[expected_names]),
    )
    assert pooled_scaled["Class"].tolist() == chong["Class"].tolist()
    assert chong_csv.read_bytes() == chong_csv_before
    pd.testing.assert_frame_equal(
        pd.read_csv(pooled_csv), pooled_scaled,
        check_exact=False, rtol=1e-14, atol=1e-14,
    )


def test_invalid_mode_fails_before_export(tmp_path):
    image = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="mode must be one of"):
        glcm_features.extract_glcm_statistics(image, mode="invalid")
    with pytest.raises(ValueError, match="mode must be one of"):
        glcm_features.extract_glcm_features(
            image, "sample.png", "class", glcm_features.PROPERTIES,
            mode="invalid",
        )
    output_dir = tmp_path / "features"
    with pytest.raises(ValueError, match="mode must be one of"):
        glcm_features.extract_features(
            tmp_path / "missing", [], output_dir, mode="invalid"
        )
    assert not output_dir.exists()
