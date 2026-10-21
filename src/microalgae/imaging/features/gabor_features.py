"""964 masked Log-Gabor descriptors following Sanchez (2019) and Bueno (2017).

The default bank is a full-resolution port of the MATLAB toolbox cited by
Sanchez, not the former Python LogGabor package. See _log_gabor_bank.py for
the exact profile and upstream provenance. Four magnitude-sum scale images
each provide 13 first-order and 19 x 3 x 4 second-order statistics.

Bueno's tables contain ambiguous signs, quartile definitions and homogeneity
formulas. This implementation retains population variance, raw-pixel
percentiles, base-2 entropy, 16-level per-ROI GLCM quantization, sum variance
centered on sum entropy, difference variance as a raw second moment, and the
squared/absolute-difference homogeneity variants. First-order mode uses a
255-bin histogram; kurtosis is Pearson (not excess). These conventions and
the chosen filter settings do not establish exact Sanchez experiment parity.
The paper's separate correlation reduction to 177 features is not performed.

Dataset loading, resizing and segmentation remain project preprocessing.
Exports are unstandardized. Regenerate existing Gabor CSVs and retrain
models when adopting this bank and the corrected first-order statistics.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from ._log_gabor_bank import _LogGaborBank, _validate_image
from ..preprocessing import segment_image, resize_with_padding


FIRST_ORDER_NAMES = (
    "mean", "mode", "min", "max", "variance", "range", "entropy",
    "q1", "q2", "q3", "iqr", "asymmetry", "kurtosis",
)
SECOND_ORDER_NAMES = (
    "energy", "variance", "contrast", "dissimilarity", "correlation",
    "autocorrelation", "entropy", "measure_corr1", "measure_corr2",
    "cluster_shade", "cluster_prominence", "max_probability", "sum_average",
    "sum_entropy", "sum_variance", "diff_entropy", "diff_variance",
    "homogeneity1", "homogeneity2",
)
DISTANCES = (1, 3, 5)
ANGLES = (0, 45, 90, 135)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FEATURE_NAMES = [
    name
    for scale in range(1, 5)
    for name in (
        [f"S{scale}_1st_{name}" for name in FIRST_ORDER_NAMES]
        + [
            f"S{scale}_2nd_{name}_d{distance}_a{angle}"
            for distance in DISTANCES for angle in ANGLES for name in SECOND_ORDER_NAMES
        ]
    )
]


def _foreground(subband, mask):
    """Validate a band and ROI, keeping genuine zero-valued foreground pixels."""
    img = np.asarray(subband)
    if img.ndim != 2 or img.size == 0 or img.dtype.kind not in "buif":
        raise ValueError("subband must be a nonempty 2-D real numeric array")
    img = img.astype(np.float64, copy=False)
    if mask is None:
        roi = np.ones(img.shape, dtype=bool)
    else:
        roi = np.asarray(mask)
        if roi.shape != img.shape or roi.dtype.kind not in "buif" or not np.isfinite(roi).all():
            raise ValueError("mask must be a finite numeric/boolean array matching the subband")
        roi = roi != 0
    if not roi.any():
        raise ValueError("Cannot extract Log-Gabor statistics from an empty foreground mask")
    foreground = img[roi]
    if not np.isfinite(foreground).all():
        raise ValueError("Foreground responses must be finite")
    return img, roi, foreground


def first_order_features(subband, bins=255, *, mask=None):
    """Compute statistics only over the ROI (all pixels when mask is omitted).

    Mode is the lowest modal bin's center (the actual value for constants).
    Constant/single-pixel ROIs use zero skewness and Pearson kurtosis as a
    finite convention, since these standardized moments are undefined there.
    """
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    _, _, flat = _foreground(subband, mask)

    # Basic stats
    mean = np.mean(flat)
    minimum = np.min(flat)
    maximum = np.max(flat)
    variance = np.var(flat)
    value_range = maximum - minimum

    # Entropy requires probabilities, not histogram probability densities.
    hist, edges = np.histogram(flat, bins=bins)
    modal_bin = np.argmax(hist)
    mode_val = minimum if minimum == maximum else (edges[modal_bin] + edges[modal_bin + 1]) / 2
    probabilities = hist[hist > 0] / hist.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))

    # Quartiles
    q1 = np.percentile(flat, 25)
    q2 = np.percentile(flat, 50)  # median
    q3 = np.percentile(flat, 75)
    iqr = q3 - q1

    # Shape statistics
    asymmetry = skew(flat) if variance > 0 else 0.0
    kurt = kurtosis(flat, fisher=False) if variance > 0 else 0.0

    features = [
        mean,
        mode_val,
        minimum,
        maximum,
        variance,
        value_range,
        entropy,
        q1,
        q2,
        q3,
        iqr,
        asymmetry,
        kurt
    ]
    if not np.isfinite(features).all():
        raise FloatingPointError("Non-finite first-order Log-Gabor statistics were calculated")
    return features


def compute_glcm_matrices(subband, levels=16, *, mask=None):
    """Build symmetric, normalized GLCMs using pairs with BOTH endpoints in ROI.

    Quantization uses only the foreground range; constant foreground maps to
    level zero. Spatial offsets match graycomatrix's distances/angles. An
    offset with no valid pairs returns an all-zero matrix (not a distribution).
    """
    if isinstance(levels, (bool, np.bool_)) or not isinstance(levels, (int, np.integer)) or levels < 2:
        raise ValueError("levels must be an integer >= 2")
    img, roi, foreground = _foreground(subband, mask)
    quantized = np.zeros(img.shape, dtype=np.int64)
    minimum, maximum = foreground.min(), foreground.max()
    if maximum > minimum:
        scaled = (foreground - minimum) / (maximum - minimum)
        quantized[roi] = np.floor(np.clip(scaled, 0, 1) * (levels - 1)).astype(np.int64)

    distances = DISTANCES
    angles = np.deg2rad(ANGLES)

    glcms = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.float64)
    height, width = img.shape
    for d, distance in enumerate(distances):
        for a, angle in enumerate(angles):
            dy = int(np.rint(distance * np.sin(angle)))
            dx = int(np.rint(distance * np.cos(angle)))
            if abs(dy) >= height or abs(dx) >= width:
                continue
            # Aligned slices preserve the original spatial relationships and
            # exclude out-of-bounds neighbors without wrapping image borders.
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            source = np.s_[y0:y1, x0:x1]
            neighbor = np.s_[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            valid = roi[source] & roi[neighbor]
            if not valid.any():
                continue
            pair_ids = quantized[source][valid] * levels + quantized[neighbor][valid]
            counts = np.bincount(pair_ids, minlength=levels * levels).reshape(levels, levels)
            counts = counts + counts.T
            glcms[:, :, d, a] = counts / counts.sum()

    return glcms

def glcm_marginals(c):
    H = c.shape[0]

    i = np.arange(H)
    j = np.arange(H)

    # marginals
    cx = np.sum(c, axis=1)  # row sum
    cy = np.sum(c, axis=0)  # col sum

    # means
    mu_x = np.sum(i * cx)
    mu_y = np.sum(j * cy)

    # std devs
    sigma_x = np.sqrt(np.sum(cx * (i - mu_x) ** 2))
    sigma_y = np.sqrt(np.sum(cy * (j - mu_y) ** 2))

    return cx, cy, mu_x, mu_y, sigma_x, sigma_y

def sum_diff_distributions(c):
    H = c.shape[0]

    cx_plus_y = np.zeros(2 * H)
    cx_minus_y = np.zeros(H)

    for i in range(H):
        for j in range(H):
            cx_plus_y[i + j] += c[i, j]
            cx_minus_y[abs(i - j)] += c[i, j]

    return cx_plus_y, cx_minus_y

def entropy_measures(c, cx, cy):
    eps = 1e-10

    # HXY1 = - sum c(i,j) log(cx(i)*cy(j))
    HXY1 = -np.sum(c * np.log2((cx[:, None] * cy[None, :]) + eps))

    # HXY2 = - sum cx(i)cy(j) log(cx(i)cy(j))
    HXY2 = -np.sum((cx[:, None] * cy[None, :]) * np.log2((cx[:, None] * cy[None, :]) + eps))

    return HXY1, HXY2


def second_order_features(subband, levels=16, *, mask=None):
    """Return 19 descriptors per offset; offsets without valid pairs use 19 zeros."""

    glcms = compute_glcm_matrices(subband, levels, mask=mask)

    features = []

    # iterate over distances and angles
    for d in range(glcms.shape[2]):      # 3 distances
        for a in range(glcms.shape[3]):  # 4 angles

            c = glcms[:, :, d, a]
            if not c.any():
                features.extend([0.0] * 19)
                continue

            H = c.shape[0]
            eps = 1e-10

            i, j = np.indices((H, H))

            cx, cy, mu_x, mu_y, sigma_x, sigma_y = glcm_marginals(c)

            cx_plus_y, cx_minus_y = sum_diff_distributions(c)

            entropy = -np.sum(c * np.log2(c + eps))

            HXY1, HXY2 = entropy_measures(c, cx, cy)

            HX = -np.sum(cx * np.log2(cx + eps))
            HY = -np.sum(cy * np.log2(cy + eps))

            energy = np.sum(c ** 2)

            variance = np.sum(((i - mu_x) ** 2) * c)

            contrast = np.sum(((i - j) ** 2) * c)

            dissimilarity = np.sum(np.abs(i - j) * c)

            correlation = (
                (np.sum(i * j * c) - mu_x * mu_y) /
                (sigma_x * sigma_y + eps)
            )

            autocorrelation = np.sum(i * j * c)

            measure_corr1 = (entropy - HXY1) / (max(HX, HY) + eps)

            measure_corr2 = np.sqrt(max(0, 1 - np.exp(-2 * (HXY2 - entropy))))

            cluster_shade = np.sum(((i + j - mu_x - mu_y) ** 3) * c)

            cluster_prominence = np.sum(((i + j - mu_x - mu_y) ** 4) * c)

            max_prob = np.max(c)

            sum_avg = np.sum(np.arange(len(cx_plus_y)) * cx_plus_y)

            sum_entropy = -np.sum(cx_plus_y * np.log2(cx_plus_y + eps))

            sum_variance = np.sum(((np.arange(len(cx_plus_y)) - sum_entropy) ** 2) * cx_plus_y)

            diff_entropy = -np.sum(cx_minus_y * np.log2(cx_minus_y + eps))

            diff_variance = np.sum((np.arange(len(cx_minus_y)) ** 2) * cx_minus_y)

            homogeneity1 = np.sum(c / (1 + (i - j) ** 2))
            homogeneity2 = np.sum(c / (1 + np.abs(i - j)))

            features.extend([
                energy,
                variance,
                contrast,
                dissimilarity,
                correlation,
                autocorrelation,
                entropy,
                measure_corr1,
                measure_corr2,
                cluster_shade,
                cluster_prominence,
                max_prob,
                sum_avg,
                sum_entropy,
                sum_variance,
                diff_entropy,
                diff_variance,
                homogeneity1,
                homogeneity2
            ])

    if not np.isfinite(features).all():
        raise FloatingPointError("Non-finite second-order Log-Gabor statistics were calculated")
    return features


def collapse_orientations(C):
    """Return four magnitude-sum bands from (H, W, 6, 4) responses."""
    C = np.asarray(C)
    if (
        C.ndim != 4 or not C.size or C.shape[2:] != (6, 4)
        or C.dtype.kind not in "buifc" or not np.isfinite(C).all()
    ):
        raise ValueError("Log-Gabor responses must be a finite numeric (H, W, 6, 4) array")
    with np.errstate(over="raise", invalid="raise"):
        return np.sum(np.abs(C.astype(np.complex128, copy=False)), axis=2)


def extract_gabor_statistics(image, mask=None, *, lg=None):
    """Calculate 964 descriptors without preprocessing or file I/O.

    Filtering requires finite real grayscale input. The default bank requires
    256 x 256 pixels; a supplied lg must expose linear_pyramid(image). The mask
    selects statistics AFTER filtering; it does not alter image intensities.
    The dataset wrapper separately preserves its existing pre-filter masking.
    """
    image = _validate_image(image)
    _, mask, _ = _foreground(image, mask)
    if lg is None:
        if image.shape != _LogGaborBank.shape:
            raise ValueError("The native Log-Gabor bank requires a 256 x 256 image; resize before filtering")
        lg = _LogGaborBank()
    if not callable(getattr(lg, "linear_pyramid", None)):
        raise ValueError("lg must expose a callable linear_pyramid(image)")
    G = collapse_orientations(lg.linear_pyramid(image))
    if mask.shape != G.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match response shape {G.shape[:2]}")
    features = []
    for scale in range(4):
        band = G[:, :, scale]
        features.extend(first_order_features(band, mask=mask))
        features.extend(second_order_features(band, mask=mask))
    result = np.asarray(features, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise FloatingPointError("Invalid Log-Gabor feature vector was calculated")
    return result


def extract_gabor_features(image, mask, image_name, subfolder_name, save_images, lg):
    """Compatibility wrapper around calculation and optional input preview."""
    try:
        features = extract_gabor_statistics(image, mask, lg=lg)
    except (ValueError, FloatingPointError) as error:
        raise type(error)(f"Cannot extract Log-Gabor features from {image_name}: {error}") from error
    if save_images:
        success_path = Path("data") / "success" / "gabor" / subfolder_name
        success_path.mkdir(parents=True, exist_ok=True)
        preview_path = success_path / f"{Path(image_name).stem}.jpg"
        preview = np.rint(np.clip(image, 0, 255)).astype(np.uint8)
        if not cv2.imwrite(str(preview_path), preview):
            raise OSError(f"Failed to save Log-Gabor preview: {preview_path}")
    return features


def extract_features(dataset_dir, class_subfolders, features_dir, save_images=False):
    """Export raw descriptors plus Class, preserving class and directory order.

    Grayscale loading, padded 256 x 256 resizing, segmentation with dilation
    size 4, and masking before filtering are unchanged. The former Python
    LogGabor parameter file is a legacy reference and is no longer loaded.
    """
    rows = []
    lg = None
    for subfolder_name in class_subfolders:
        subfolder_path = Path(dataset_dir) / subfolder_name
        subfolder_images = [
            name for name in os.listdir(subfolder_path)
            if (subfolder_path / name).is_file() and Path(name).suffix.lower() in IMAGE_EXTENSIONS
        ]
        print(f"Class: {subfolder_name} ({len(subfolder_images)} images)")
        for image_name in subfolder_images:
            image_path = subfolder_path / image_name
            try:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                image = resize_with_padding(image, (256, 256))
                mask, _, _ = segment_image(image, dilation_size=4)
                _, mask, _ = _foreground(image, mask)
                masked_image = image * mask.astype(image.dtype)
                if lg is None:
                    lg = _LogGaborBank()
                features = extract_gabor_features(
                    masked_image, mask, image_name, subfolder_name, save_images, lg
                )
            except (ValueError, FloatingPointError, OSError, cv2.error) as error:
                raise type(error)(f"Failed to extract Log-Gabor features from {image_path}: {error}") from error
            rows.append([*features, subfolder_name])

    if not rows:
        raise ValueError("No supported images found for Log-Gabor feature extraction")
    df = pd.DataFrame(rows, columns=[*FEATURE_NAMES, "Class"])
    export_path = Path(features_dir) / "Gabor_features.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(export_path, index=False)
    return df
