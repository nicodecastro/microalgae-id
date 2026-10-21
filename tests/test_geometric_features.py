"""Reference-equivalence and failure-handling tests, independent of preprocessing."""

import cv2
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from microalgae.imaging.features import geometric_features as geometric


def reference_features(contour_image):
    """The supplied reference's nine formulas and aggregation for valid contours."""
    contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for contour in contours:
        if len(contour) < 5:
            continue
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        _, (w, h), _ = cv2.minAreaRect(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        rect = cv2.boundingRect(contour)
        _, (MA, ma), _ = cv2.fitEllipse(contour)
        rows.append([
            np.sqrt(1 - (MA / ma) ** 2) if ma else 0,
            area / hull_area, max(w, h), area, perimeter,
            area / hull_area if hull_area else 0,
            area / (rect[2] * rect[3]) if rect[2] * rect[3] else 0,
            w / h if w > h else h / w,
            4 * np.pi * area / perimeter ** 2,
        ])
    # Select the requested five features from the original nine, in output order.
    return np.mean(rows, axis=0)[[3, 4, 0, 7, 8]].tolist() if rows else None


def shape_image(kind):
    image = np.zeros((180, 240), dtype=np.uint8)
    if kind == "concave":
        points = np.array([[20, 20], [110, 20], [80, 70], [120, 120], [20, 100], [40, 60]])
        cv2.fillPoly(image, [points], 255)
    else:
        cv2.ellipse(image, (65, 70), (40, 22), 27, 0, 360, 255, -1)
        if kind == "hole":
            cv2.circle(image, (65, 70), 10, 0, -1)
        elif kind == "multiple":
            cv2.circle(image, (175, 115), 15, 255, -1)
    return image


@pytest.mark.parametrize("kind", ["ellipse", "concave", "hole", "multiple"])
def test_selected_five_descriptors_match_supplied_reference(kind):
    image = shape_image(kind)
    actual = geometric.extract_geometric_features(image)
    np.testing.assert_allclose(actual, reference_features(image), rtol=1e-12, atol=1e-12)
    assert len(actual) == 5
    assert actual[3] >= 1


def test_multiple_contours_are_averaged_equally_not_by_area():
    image = shape_image("multiple")
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    actual = geometric.extract_geometric_features(image)
    assert actual[0] == pytest.approx(np.mean(areas))
    assert actual[0] != pytest.approx(max(areas))


def test_ellipse_axis_order_is_robust(monkeypatch):
    image = shape_image("ellipse")
    original = geometric.extract_geometric_features(image)
    fit_ellipse = cv2.fitEllipse
    def reversed_axes(contour):
        center, (a, b), angle = fit_ellipse(contour)
        return center, (b, a), angle
    monkeypatch.setattr(cv2, "fitEllipse", reversed_axes)
    np.testing.assert_allclose(geometric.extract_geometric_features(image), original)


def test_degenerate_and_short_contours_do_not_contaminate_mean():
    valid = cv2.ellipse2Poly((50, 50), (25, 12), 0, 0, 360, 15).reshape(-1, 1, 2)
    line = np.array([[[i, i]] for i in range(6)], dtype=np.int32)
    triangle = np.array([[[0, 0]], [[10, 0]], [[0, 10]]], dtype=np.int32)
    np.testing.assert_allclose(
        geometric.extract_features_from_contours([line, valid, triangle]),
        geometric.extract_features_from_contours([valid]),
    )
    assert geometric.extract_features_from_contours([line, triangle]) is None


def test_empty_image_and_four_point_rectangle_return_none():
    image = np.zeros((100, 100), dtype=np.uint8)
    assert geometric.extract_geometric_features(image) is None
    cv2.rectangle(image, (10, 10), (90, 90), 255, -1)
    # Preserves the reference's <5-point rejection with CHAIN_APPROX_SIMPLE.
    assert geometric.extract_geometric_features(image) is None


def test_existing_preprocessing_is_unchanged():
    image = cv2.cvtColor(shape_image("multiple"), cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prepared = cv2.Canny(cv2.bilateralFilter(gray, 40, 1, 25), 100, 170)
    np.testing.assert_allclose(
        geometric.extract_features_from_image(image, "test.png", "alga"),
        reference_features(prepared),
    )


def test_invalid_images_raise_clear_errors():
    with pytest.raises(ValueError, match="BGR"):
        geometric.extract_features_from_image(None, "missing.png", "alga")
    with pytest.raises(ValueError, match="uint8"):
        geometric.extract_geometric_features(np.zeros((8, 8), dtype=float))


def test_batch_export_scaling_and_failed_images(tmp_path):
    raw = tmp_path / "raw" / "alga"
    raw.mkdir(parents=True)
    first = shape_image("ellipse")
    second = shape_image("multiple")
    assert cv2.imwrite(str(raw / "b.png"), second)
    assert cv2.imwrite(str(raw / "a.png"), first)
    assert cv2.imwrite(str(raw / "blank.png"), np.zeros_like(first))
    (raw / "corrupt.jpg").touch()
    (raw / "ignored.txt").touch()
    with pytest.warns(UserWarning, match="Skipping") as warnings_seen:
        unscaled = geometric.extract_features(tmp_path / "raw", ["alga"], tmp_path / "unscaled", standardize=False)
    assert len(warnings_seen) == 2
    assert list(unscaled.columns) == [
        "Area", "Perimeter", "Eccentricity", "AspectRatio", "Circularity", "image_id", "Class",
    ]
    assert unscaled.shape == (2, 7)
    assert unscaled["image_id"].tolist() == ["alga/a.png", "alga/b.png"]
    expected = [
        geometric.extract_features_from_image(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), "test.png", "alga")
        for img in (first, second)
    ]
    np.testing.assert_allclose(unscaled[geometric.FEATURE_NAMES], expected)
    with pytest.warns(UserWarning, match="Skipping"):
        scaled = geometric.extract_features(tmp_path / "raw", ["alga"], tmp_path / "scaled")
    np.testing.assert_allclose(scaled[geometric.FEATURE_NAMES], StandardScaler().fit_transform(expected))
    pd.testing.assert_frame_equal(scaled, pd.read_csv(tmp_path / "scaled" / "Geometric_shape_features.csv"))


def test_all_failed_dataset_does_not_export(tmp_path):
    (tmp_path / "raw" / "alga").mkdir(parents=True)
    assert cv2.imwrite(str(tmp_path / "raw" / "alga" / "blank.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="No valid"):
        geometric.extract_features(tmp_path / "raw", ["alga"], tmp_path / "out")
    assert not (tmp_path / "out" / "Geometric_shape_features.csv").exists()


def test_success_and_failure_previews(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    valid = cv2.cvtColor(shape_image("ellipse"), cv2.COLOR_GRAY2BGR)
    assert geometric.extract_features_from_image(valid, "good.png", "alga", True) is not None
    assert geometric.extract_features_from_image(np.zeros_like(valid), "blank.png", "alga", True) is None
    assert cv2.imread(str(tmp_path / "data" / "success" / "geometric" / "alga" / "good.jpg")) is not None
    assert cv2.imread(str(tmp_path / "data" / "failed" / "geometric" / "alga" / "blank.jpg")) is not None
