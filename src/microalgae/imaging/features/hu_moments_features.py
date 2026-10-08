"""Seven mean Hu moments per image, following the supplied Chong reference.

For every external contour, calculate cv2.moments and cv2.HuMoments, flatten
the seven values, then take an unweighted mean across contours. An image with
no contours yields seven zeros. No contour filtering or log transform is used.

The image wrapper retains the existing grayscale, bilateral, and adaptive
threshold steps. Use extract_hu_moments_from_binary to inspect the descriptor
steps independently of that preprocessing.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


FEATURE_NAMES = [f"Hu_{index}" for index in range(1, 8)]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def extract_hu_moments_from_binary(binary_image):
    """Return the reference's seven descriptors from an 8-bit contour image."""
    binary_image = np.asarray(binary_image)
    if binary_image.ndim != 2 or binary_image.size == 0 or binary_image.dtype != np.uint8:
        raise ValueError("binary_image must be a nonempty 2-D uint8 array")

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return [0.0] * 7

    # Retain every contour, including degenerate ones, as in the reference.
    hu_values = [cv2.HuMoments(cv2.moments(contour)).flatten() for contour in contours]
    return np.mean(hu_values, axis=0).tolist()


def extract_hu_moments(image_path, image_name, subfolder_name, save_images=False):
    """Read and preprocess one image, then calculate mean Hu moments."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Keep preprocessing identical to the original extractor and reference.
    blurred = cv2.bilateralFilter(image, 40, 1, 25)
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )
    features = extract_hu_moments_from_binary(thresholded)

    if save_images:
        status = "failed" if cv2.countNonZero(thresholded) == 0 else "success"
        output_dir = Path("data") / status / "hu_moments" / subfolder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(image_name).stem}.jpg"
        if not cv2.imwrite(str(output_path), thresholded):
            raise OSError(f"Failed to save thresholded image: {output_path}")

    return features


def extract_features(
    dataset_dir, class_subfolders, features_dir, save_images=False, *,
    standardize=True,
):
    """Export Hu_1..Hu_7 plus Class; return the DataFrame.

    The default keeps the reference's dataset-wide StandardScaler and the
    existing CSV filename. For classifier evaluation, use standardize=False
    and fit a scaler only on training data, inside each validation fold.
    """
    rows = []
    for label in class_subfolders:
        class_dir = Path(dataset_dir) / label
        paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"Class: {label} ({len(paths)} images)")
        for path in paths:
            rows.append([
                *extract_hu_moments(path, path.name, label, save_images), label
            ])

    if not rows:
        raise ValueError("No supported images found for Hu moments extraction")

    frame = pd.DataFrame(rows, columns=[*FEATURE_NAMES, "Class"])
    if standardize:
        from sklearn.preprocessing import StandardScaler

        frame[FEATURE_NAMES] = StandardScaler().fit_transform(frame[FEATURE_NAMES])

    output_path = Path(features_dir) / "Hu_moments_nonminmax_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
