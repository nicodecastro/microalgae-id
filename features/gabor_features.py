from pathlib import Path
import cv2
import numpy as np
from scipy.stats import skew, kurtosis, mode
import pandas as pd
import os
from preprocessing import segment_image, resize_with_padding

def _foreground(subband, mask):
    """Validate a band and ROI, keeping genuine zero-valued foreground pixels."""
    img = np.asarray(subband, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        raise ValueError("subband must be a nonempty 2-D array")
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


# Compute first order statistics for each sub-band
def first_order_features(subband, bins=256, *, mask=None):
    """Compute statistics only over the ROI (all pixels when mask is omitted).

    Constant/single-pixel ROIs use zero skewness and excess kurtosis as a
    finite convention, since these standardized moments are undefined there.
    """
    _, _, flat = _foreground(subband, mask)

    # Basic stats
    mean = np.mean(flat)
    mode_val = mode(flat, keepdims=False).mode
    minimum = np.min(flat)
    maximum = np.max(flat)
    variance = np.var(flat)
    value_range = maximum - minimum

    # Entropy requires probabilities, not histogram probability densities.
    hist, _ = np.histogram(flat, bins=bins)
    probabilities = hist[hist > 0] / hist.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))

    # Quartiles
    q1 = np.percentile(flat, 25)
    q2 = np.percentile(flat, 50)  # median
    q3 = np.percentile(flat, 75)
    iqr = q3 - q1

    # Shape statistics
    asymmetry = skew(flat) if variance > 0 else 0.0
    kurt = kurtosis(flat) if variance > 0 else 0.0

    return [
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
    
# Compute GLCMs for each sub-band (Bueno et al., 2017 as defined in Sanchez et al., 2019)
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

    distances = [1, 3, 5] # 1, 3, 5 pixels
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135

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

# # Compute features from each GLCM of the sub-bands (Bueno et al., 2017 as defined in Sanchez et al., 2019)
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

    return features

def collapse_orientations(C):
    # C: (H, W, 6, 4)
    return np.sum(np.abs(C), axis=2)  # (H, W, 4)
    
def extract_gabor_features(image, mask, image_name, subfolder_name, save_images, lg):
    # Reject failed segmentation before doing the expensive filter transform.
    _, mask, _ = _foreground(image, mask)
    # Step 1: Log-Gabor responses
    C = lg.linear_pyramid(image)  # (H, W, 6, 4)

    # Step 2: collapse orientations → 4 bands
    G = collapse_orientations(C)  # (H, W, 4)

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != G.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match response shape {G.shape[:2]}."
        )

    # Pass the ROI explicitly: zeroing background alone would still count it.

    features = []

    # Step 3: feature extraction per scale band
    for s in range(4):

        subband = G[:, :, s]

        f1 = first_order_features(subband, mask=mask)
        f2 = second_order_features(subband, mask=mask)

        # FIRST ORDER (13)
        features.extend(f1)

        # SECOND ORDER (228)
        features.extend(f2)

        # print("Band", s, "first-order:", len(f1), "second-order:", len(f2))

    if save_images:
        success_path = Path(f"data/success/efd/{subfolder_name}")
        success_path.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
        cv2.imwrite(str(os.path.join(success_path, f"{Path(image_name).stem}.jpg")), image)

    return np.array(features)
  
def extract_features(dataset_dir, class_subfolders, features_dir, save_images=False):
    from LogGabor import LogGabor

    param_path = Path(__file__).parent.parent / 'utils' / 'default_param.py'
    lg = LogGabor(str(param_path))

    # Initialize an empty list to store the feature data
    feature_data = []
    labels_data = []

    feature_names = []

    # 13 features
    first_order_names = [
        "mean", "mode", "min", "max", "variance",
        "range", "entropy", "q1", "q2", "q3",
        "iqr", "asymmetry", "kurtosis"
    ]

    # SECOND-ORDER (19 features × 3 distances × 4 angles × 4 scales)
    second_order_names = [
        "energy",
        "variance",
        "contrast",
        "dissimilarity",
        "correlation",
        "autocorrelation",
        "entropy",
        "measure_corr1",
        "measure_corr2",
        "cluster_shade",
        "cluster_prominence",
        "max_probability",
        "sum_average",
        "sum_entropy",
        "sum_variance",
        "diff_entropy",
        "diff_variance",
        "homogeneity1",
        "homogeneity2"
    ]

    distances = [1, 3, 5]
    angles = [0, 45, 90, 135]

    for s in range(1, 5):
        # Match extraction order: 13 first-order + 228 second-order per scale.
        for name in first_order_names:
            feature_names.append(f"S{s}_1st_{name}")
        for d in distances:
            for a in angles:
                for feat in second_order_names:
                    feature_names.append(f"S{s}_2nd_{feat}_d{d}_a{a}")
                    
    # Loop through the subfolders and extract features from each image
    for subfolder_name in class_subfolders:
        subfolder_path = os.path.join(dataset_dir, subfolder_name)
        subfolder_images = os.listdir(subfolder_path)
        print(f"Class: {subfolder_name} ({len(subfolder_images)} images)")

        # Loop over the images in the subfolder and extract the features
        for image_name in subfolder_images:
            # Load the image
            image_path = os.path.join(subfolder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            image = resize_with_padding(image, (256, 256))
            
            # Preprocess images based on Sanchez et al. (2024)
            mask, _, _ = segment_image(image, dilation_size=4)
            
            masked_image = image * mask.astype(image.dtype)
            
            # Extract the features from the image
            features = extract_gabor_features(
                masked_image, mask, image_name, subfolder_name, save_images, lg
            )
            feature_data.append(features)
            labels_data.append(subfolder_name)
            
    # Convert the features and labels to a pandas dataframe
    df = pd.DataFrame(feature_data, columns=feature_names)
    df['Class'] = labels_data

    # Print the first few rows of the dataframe
    print(df)

    # Export the DataFrame to a CSV file
    export_path = Path(features_dir) / "Gabor_features.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
    df.to_csv(export_path, index=False)
