"""Whole-image GLCM descriptors with Chong-code and orientation-pooled modes.

The Chong-code mode computes three distances and four angles but takes only
the first value of each graycoprops result: distance 1, angle 0. The pooled
mode averages each property over angles separately at each distance. Neither
mode applies the Log-Gabor foreground mask.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

from ..dataset import ExtractionFailureReport, IMAGE_EXTENSIONS, image_id
from ..preprocessing.configured import prepare_feature_input, save_prepared_input, validate_preprocessing

PROPERTIES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
DISTANCES = [1, 3, 5]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_MODES = ("chong_code", "orientation_pooled")


def _validate_mode(mode):
    if mode not in GLCM_MODES:
        raise ValueError(f"mode must be one of {GLCM_MODES}; got {mode!r}")


def extract_glcm_statistics(image, properties=PROPERTIES, *, mode="chong_code"):
    """Return selected properties from an already prepared grayscale image."""
    _validate_mode(mode)
    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("image must be a nonempty 2-D uint8 array")

    glcm = graycomatrix(
        image, distances=DISTANCES, angles=ANGLES, levels=256,
        symmetric=True, normed=True,
    )
    features = []
    for prop in properties:
        values = graycoprops(glcm, prop)
        if mode == "chong_code":
            features.append(values.ravel()[0])
        else:
            features.extend(values.mean(axis=1).tolist())
    return features


def extract_glcm_features(
    image, image_name, subfolder_name, properties, save_images=False, *,
    mode="chong_code",
):
    """Keep the existing public interface and optional image preview."""
    features = extract_glcm_statistics(image, properties, mode=mode)

    if save_images:
        output_dir = Path("data") / "success" / "glcm" / subfolder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(image_name).stem}.jpg"
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Failed to save GLCM image: {output_path}")

    return features


def extract_features(
    dataset_dir, class_subfolders, features_dir, save_images=False, *,
    standardize=True, mode="chong_code", preprocessing=None,
):
    """Export selected GLCM columns, image_id, and Class in directory order.

    The default reproduces the Chong reference and its dataset-wide scaler.
    Orientation pooling exports fifteen features to a separate CSV.
    Use ``standardize=False`` to export raw GLCM properties.
    preprocessing opts into shared preparation without the legacy 512 resize.
    """
    _validate_mode(mode)
    if preprocessing is not None:
        preprocessing = validate_preprocessing(preprocessing, "glcm")
    feature_names = (
        PROPERTIES if mode == "chong_code"
        else [f"{prop}_d{distance}" for prop in PROPERTIES for distance in DISTANCES]
    )
    filename = (
        "GLCM_features.csv" if mode == "chong_code"
        else "GLCM_orientation_pooled_features.csv"
    )
    output_path = Path(features_dir) / filename
    failures = ExtractionFailureReport(output_path, dataset_dir)
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
            with failures.track(image_path, label):
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE if preprocessing is None else cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                if preprocessing is None:
                    resized = cv2.resize(image, (512, 512))
                    features = extract_glcm_features(
                        resized, image_name, label, PROPERTIES, save_images, mode=mode
                    )
                else:
                    prepared = prepare_feature_input(image, preprocessing, "glcm")
                    features = extract_glcm_statistics(prepared.image, mode=mode)
                    if save_images:
                        save_prepared_input(prepared, features_dir, "glcm", image_id(image_path, dataset_dir))
                rows.append([*features, image_id(image_path, dataset_dir), label])

    if not rows:
        raise ValueError("No supported images found for GLCM feature extraction")

    frame = pd.DataFrame(rows, columns=[*feature_names, "image_id", "Class"])
    if standardize:
        frame[feature_names] = StandardScaler().fit_transform(frame[feature_names])

    print(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
