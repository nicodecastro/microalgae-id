"""Uniform LBP histograms following the supplied Chong reference.

The descriptor uses 8 points, radius 3, ten histogram bins, and the
reference's ``hist.sum() + 1e-7`` normalization. The image wrapper keeps
grayscale loading and 512x512 resizing unchanged.
"""

from pathlib import Path

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from ..dataset import image_id
from ._export import export_feature_table
from ..preprocessing.configured import prepare_feature_input, save_prepared_input, validate_preprocessing

FEATURE_NAMES = [f"feature_{index}" for index in range(10)]


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
    standardize=True, preprocessing=None,
):
    """Export ten LBP columns, image_id, and Class in input directory order.

    The default reproduces the reference's dataset-wide StandardScaler.
    Use ``standardize=False`` to export raw histograms.
    preprocessing opts into shared preparation without the legacy 512 resize.
    """
    if preprocessing is not None:
        preprocessing = validate_preprocessing(preprocessing, "lbp")

    def extract_image(path, label):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if preprocessing is None else cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        if preprocessing is None:
            resized = cv2.resize(image, (512, 512))
            hist = extract_lbp_features(
                resized, path.name, label, method="uniform", save_images=save_images
            )
        else:
            prepared = prepare_feature_input(image, preprocessing, "lbp")
            hist = extract_lbp_histogram(prepared.image)
            if save_images:
                save_prepared_input(prepared, features_dir, "lbp", image_id(path, dataset_dir))
        return hist

    frame = export_feature_table(
        dataset_dir, class_subfolders, Path(features_dir) / "LBP_features.csv",
        FEATURE_NAMES, extract_image,
        empty_message="No supported images found for LBP feature extraction",
        standardize=standardize,
        sort_paths=False,
    )
    print(frame)
    return frame
