"""Thirty harmonic amplitudes following Sanchez et al. (2019), equation 8.

Raw amplitudes retain size in processed pixels. Optional shape normalization
uses PyEFD's first-harmonic normalization before calculating amplitudes. This
is an extension: the paper does not specify its size-normalization procedure.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyefd import elliptic_fourier_descriptors

from ..preprocessing import segment_image


NUM_HARMONICS = 30
FEATURE_NAMES = [f"efd_amp{index}" for index in range(1, NUM_HARMONICS + 1)]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def extract_efd_amplitudes(contour, *, normalize_shape=False):
    """Return 30 amplitudes from a finite (N, 2) contour with nonzero area."""
    points = np.asarray(contour, dtype=np.float64)
    if (
        points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
        or not np.isfinite(points).all()
    ):
        raise ValueError("contour must be a finite (N, 2) array with at least three points")

    points = points[np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]]
    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("contour must enclose a nonzero area")

    relative = points - points[0]
    twice_area = np.sum(
        relative[:, 0] * np.roll(relative[:, 1], -1)
        - relative[:, 1] * np.roll(relative[:, 0], -1)
    )
    if not np.isfinite(twice_area) or twice_area == 0:
        raise ValueError("contour must enclose a nonzero area")

    closed = np.vstack([points, points[0]])
    coefficients = elliptic_fourier_descriptors(
        closed, order=NUM_HARMONICS, normalize=normalize_shape
    )
    amplitudes = 0.5 * np.sqrt(np.sum(coefficients ** 2, axis=1))
    if not np.isfinite(amplitudes).all():
        raise ValueError("Non-finite EFD amplitudes were calculated")
    return amplitudes.tolist()


def extract_efd_features(
    image, image_name, subfolder_name, save_images=False, *, normalize_shape=False,
):
    """Extract from the largest external contour of a bool, 0/1, or 0/255 mask."""
    mask = np.asarray(image)
    if (
        mask.ndim != 2 or not mask.size or mask.dtype.kind not in "buif"
        or not np.isfinite(mask).all()
        or not (np.isin(mask, [0, 1]).all() or np.isin(mask, [0, 255]).all())
    ):
        raise ValueError(f"Expected a nonempty binary mask (bool, 0/1, or 0/255): {image_name}")
    mask_uint8 = (mask != 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError(f"No foreground contour found: {image_name}")
    largest = max(contours, key=cv2.contourArea)
    try:
        features = extract_efd_amplitudes(
            largest.reshape(-1, 2), normalize_shape=normalize_shape
        )
    except ValueError as error:
        raise ValueError(f"Cannot extract EFD features from {image_name}: {error}") from error

    if save_images:
        output_dir = Path("data") / "success" / "efd" / subfolder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(image_name).stem}.jpg"
        if not cv2.imwrite(str(output_path), mask_uint8):
            raise OSError(f"Failed to save EFD mask: {output_path}")
    return features


def extract_features(
    dataset_dir, class_subfolders, features_dir, save_images=False, *,
    normalize_shape=False,
):
    """Export 30 amplitudes plus Class, preserving class and directory order.

    Both variants use the existing segmentation and retain unstandardized
    feature columns. Shape-normalized amplitudes are saved to a separate CSV.
    """
    rows = []
    for label in class_subfolders:
        class_dir = Path(dataset_dir) / label
        image_names = [
            name for name in os.listdir(class_dir)
            if (class_dir / name).is_file()
            and Path(name).suffix.lower() in IMAGE_EXTENSIONS
        ]
        print(f"Class: {label} ({len(image_names)} images)")
        for image_name in image_names:
            image_path = class_dir / image_name
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            mask, _, _ = segment_image(image, dilation_size=4)
            features = extract_efd_features(
                mask, image_name, label, save_images, normalize_shape=normalize_shape
            )
            rows.append([*features, label])

    if not rows:
        raise ValueError("No supported images found for EFD feature extraction")

    frame = pd.DataFrame(rows, columns=[*FEATURE_NAMES, "Class"])
    print(frame)
    filename = (
        "EFD_shape_normalized_features.csv" if normalize_shape else "EFD_features.csv"
    )
    output_path = Path(features_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
