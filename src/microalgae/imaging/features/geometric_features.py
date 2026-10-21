"""Geometric descriptors matching the supplied Chong reference's contour stage.

Keep Area, Perimeter, Eccentricity, AspectRatio, and Circularity in that order.
Their formulas, RETR_EXTERNAL / CHAIN_APPROX_SIMPLE, minimum of five contour
points, and unweighted contour averaging follow the supplied reference.

Preprocessing in extract_features_from_image remains grayscale -> bilateral
(40, 1, 25) -> Canny (100, 170). Use extract_geometric_features on an already
prepared contour image to evaluate feature extraction independently of that
preprocessing. With identical nondegenerate contours, raw descriptors match
the reference to floating-point precision.

Degenerate/non-finite contours are excluded, and images without valid contours
return None. Label-dependent interpolation is unnecessary and is not performed.
Dataset export retains z-score standardization by default for compatibility;
use standardize=False when fitting scaling inside training/CV folds.
"""

from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd

from ..dataset import ExtractionFailureReport, IMAGE_EXTENSIONS, image_id

FEATURE_NAMES = [
    "Area", "Perimeter", "Eccentricity", "AspectRatio", "Circularity",
]


def _contour_features(contour):
    """Return five finite descriptors, or None for an unusable contour."""
    if len(contour) < 5 or not np.isfinite(contour).all():
        return None

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    _, (width, height), _ = cv2.minAreaRect(contour)
    denominators = np.array([area, perimeter, width, height])
    if not np.isfinite(denominators).all() or np.any(denominators <= 0):
        return None

    try:
        _, axes, _ = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    minor, major = sorted(axes)
    if not np.isfinite([minor, major]).all() or minor <= 0:
        return None
    eccentricity = np.sqrt(np.clip(1.0 - (minor / major) ** 2, 0.0, 1.0))
    values = np.array([
        area,
        perimeter,
        eccentricity,
        max(width, height) / min(width, height),
        4.0 * np.pi * area / perimeter ** 2,
    ])
    return values if np.isfinite(values).all() else None


def extract_features_from_contours(contours):
    """Average valid contours equally, as in the reference; return None if empty.

    This retains the reference's five-point rule even for shapes that could
    otherwise supply area/perimeter but cannot support fitEllipse.
    """
    values = []
    for contour in contours:
        features = _contour_features(contour)
        if features is not None:
            values.append(features)
    return np.mean(values, axis=0).tolist() if values else None


def extract_geometric_features(contour_image):
    """Extract from an already prepared 2-D uint8 image without preprocessing.

    OpenCV treats any nonzero pixel as foreground, matching the reference's
    findContours call. No extra thresholding, resizing, or filtering is applied.
    """
    contour_image = np.asarray(contour_image)
    if contour_image.ndim != 2 or not contour_image.size or contour_image.dtype != np.uint8:
        raise ValueError("contour_image must be a nonempty 2-D uint8 array")
    contours, _ = cv2.findContours(
        contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return extract_features_from_contours(contours)


def extract_features_from_image(image, image_name, subfolder_name, save_images=False):
    """Preserve the existing BGR preprocessing wrapper and return raw features."""
    if (
        image is None or image.ndim != 3 or image.shape[2] != 3
        or image.size == 0 or image.dtype != np.uint8
    ):
        raise ValueError(f"Expected a nonempty uint8 BGR image: {image_name}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 40, 1, 25)
    canny = cv2.Canny(blurred, 100, 170)
    features = extract_geometric_features(canny)

    if save_images:
        status = "success" if features is not None else "failed"
        output_dir = Path("data") / status / "geometric" / subfolder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(image_name).stem}.jpg"
        if not cv2.imwrite(str(output_path), canny):
            raise OSError(f"Failed to save contour image: {output_path}")
    return features


def extract_features(
    dataset_dir, class_subfolders, features_dir, save_images=False, *,
    standardize=True,
):
    """Export five geometric columns, image_id, and Class; return the DataFrame.

    Images are visited in supplied class order, then sorted filename order.
    Unreadable images and images with no valid contours are warned about and
    skipped, rather than fabricating descriptors from class labels. Consequently
    successful rows need not align positionally with another feature table.

    standardize=True reproduces the reference's dataset-wide StandardScaler.
    For model evaluation, choose False and fit a scaler on training folds only.
    An entirely unsuccessful extraction raises before writing a CSV.
    """
    output_path = Path(features_dir) / "Geometric_shape_features.csv"
    failures = ExtractionFailureReport(output_path, dataset_dir)
    rows = []
    for label in class_subfolders:
        class_dir = Path(dataset_dir) / label
        paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"Class: {label} ({len(paths)} images)")
        for path in paths:
            with failures.track(path, label):
                image = cv2.imread(str(path))
                if image is None:
                    failures.record(path, label, ValueError(f"Skipping unreadable image: {path}"))
                    warnings.warn(f"Skipping unreadable image: {path}", stacklevel=2)
                    continue
                features = extract_features_from_image(image, path.name, label, save_images)
                if features is None:
                    failures.record(path, label, ValueError(f"Skipping image with no valid contours: {path}"))
                    warnings.warn(f"Skipping image with no valid contours: {path}", stacklevel=2)
                    continue
                rows.append([*features, image_id(path, dataset_dir), label])

    if not rows:
        raise ValueError("No valid geometric features were extracted; CSV was not written")
    frame = pd.DataFrame(rows, columns=[*FEATURE_NAMES, "image_id", "Class"])
    if standardize:
        from sklearn.preprocessing import StandardScaler

        frame[FEATURE_NAMES] = StandardScaler().fit_transform(frame[FEATURE_NAMES])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
