"""Uniform LBP histograms following the supplied Chong reference.

The descriptor uses 8 points, radius 3, ten histogram bins, and the
reference's ``hist.sum() + 1e-7`` normalization. The image wrapper keeps
grayscale loading and 512x512 resizing unchanged.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [f"feature_{index}" for index in range(10)]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def extract_lbp_histogram(image, method="uniform", radius=3, n_points=8):
    """Calculate the reference histogram from an already prepared image."""
    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("image must be a nonempty 2-D uint8 array")

    lbp = local_binary_pattern(image, n_points, radius, method=method)
    hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    return hist


def extract_lbp_features(
    image, image_name, subfolder_name, method="uniform", radius=3,
    n_points=8, save_images=False,
):
    """Keep the existing public interface and optional image preview."""
    hist = extract_lbp_histogram(image, method=method, radius=radius, n_points=n_points)

    if save_images:
        output_dir = Path("data") / "success" / "lbp" / subfolder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(image_name).stem}.jpg"
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Failed to save LBP image: {output_path}")

    return hist


def extract_features(
    dataset_dir, class_subfolders, features_dir, save_images=False, *,
    standardize=True,
):
    """Export ten LBP columns plus Class in the input directory order.

    The default reproduces the reference's dataset-wide StandardScaler.
    Use ``standardize=False`` to export raw histograms.
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
            resized = cv2.resize(image, (512, 512))
            hist = extract_lbp_features(
                resized, image_name, label, method="uniform", save_images=save_images
            )
            rows.append([*hist, label])

    if not rows:
        raise ValueError("No supported images found for LBP feature extraction")

    frame = pd.DataFrame(rows, columns=[*FEATURE_NAMES, "Class"])
    if standardize:
        frame[FEATURE_NAMES] = StandardScaler().fit_transform(frame[FEATURE_NAMES])

    print(frame)
    output_path = Path(features_dir) / "LBP_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
