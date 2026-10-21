"""Four phase-congruency descriptors from Sanchez et al. (2019), pp. 8-9.

Compute mean/std of the maximum (M) and minimum (m) moment images using
Phasepack's port of Kovesi's phasecong3. Install requirements.txt.
See https://doi.org/10.7717/peerj.6770 and
https://github.com/alimuldal/phasepack/blob/master/phasepack/phasecong.py.

The paper does not specify PC filter settings, the std denominator, or an
explicit PC statistics mask. Defaults here are 4 scales / 6 orientations,
population std (ddof=0), and all image pixels. Four scales follows Kovesi's
cited MATLAB code, not a confirmed Sanchez setting:
https://www.peterkovesi.com/matlabfns/PhaseCongruency/phasecong3.m.
The paper's explicit 4-scale / 6-orientation bank describes Gabor features.
These choices implement the described descriptor, not verified numerical
reproduction of the authors' experiment. Phasepack retains Kovesi's covariance
normalization and epsilon; its minimum moment can be slightly negative.

Single image: phase_congruency_features(gray, mask=roi).
Dataset: extract_features("data/raw", ["n_palea"], "data/features").
Use nscale=5 (or phasecong_parameters={"nscale": 5} for datasets) to retain
the previous Phasepack scale default. Adopting four scales requires regenerating
existing PC CSVs and retraining models that use them.
Dataset resizing defaults to this project's padded 256 x 256 setup; pass
resize=None for native resolution. No thresholding or edge detection is applied.
Exported descriptors are unstandardized: fit feature scaling on training data.
"""

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from pathlib import Path

import numpy as np


pc_feature_names = ["PC_mean_M", "PC_std_M", "PC_mean_m", "PC_std_m"]


def _validate_phasecong_parameters(parameters):
    if not isinstance(parameters, Mapping):
        raise ValueError("phasecong_parameters must be a mapping")
    defaults = dict(
        nscale=4, norient=6, minWaveLength=3, mult=2.1, sigmaOnf=0.55,
        k=2, cutOff=0.5, g=10, noiseMethod=-1,
    )
    unknown = parameters.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown phase-congruency parameters: {', '.join(sorted(map(str, unknown)))}")
    settings = {**defaults, **parameters}
    for name in ("nscale", "norient"):
        value = settings[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    bounds = {
        "minWaveLength": (lambda v: v >= 2, ">= 2"),
        "mult": (lambda v: v > 1, "> 1"),
        "sigmaOnf": (lambda v: 0 < v < 1, "in (0, 1)"),
        "k": (lambda v: v >= 0, ">= 0"),
        "cutOff": (lambda v: 0 <= v <= 1, "in [0, 1]"),
        "g": (lambda v: v > 0, "> 0"),
        "noiseMethod": (lambda v: v in (-1, -2) or v >= 0, "equal to -1, -2, or >= 0"),
    }
    for name, (in_range, requirement) in bounds.items():
        value = settings[name]
        try:
            valid = not isinstance(value, (bool, np.bool_)) and isinstance(value, Real) and isfinite(value)
        except OverflowError:
            valid = False
        if not valid or not in_range(value):
            raise ValueError(f"{name} must be a finite real scalar {requirement}")
    return settings


def phase_congruency_moments(image, *, nscale=4, norient=6, **parameters):
    """Return float64 M and m maps from a finite, 2-D grayscale image.

    Four scales follows Kovesi's cited code; nscale=5 restores the old default.
    Remaining keyword arguments go to phasepack.phasecong. Its defaults are
    minWaveLength=3, mult=2.1, sigmaOnf=0.55, k=2, cutOff=0.5, g=10,
    noiseMethod=-1. Intensities retain their supplied scale (normally 0..255).
    """
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError("image must be a 2-D grayscale array at least 2 x 2")
    if image.dtype.kind not in "buif" or not np.isfinite(image).all():
        raise ValueError("image must contain finite real numeric values")
    settings = _validate_phasecong_parameters(dict(nscale=nscale, norient=norient, **parameters))

    from phasepack import phasecong

    image = image.astype(np.float64, copy=False)
    # No spatial variation means no edges/corners. Phasepack 1.5 divides by
    # zero for constant images; return the physically meaningful zero maps.
    if np.all(image == image.flat[0]):
        return np.zeros_like(image), np.zeros_like(image)

    with np.errstate(divide="ignore", invalid="ignore"):
        M, m, _, _, pc, responses, _ = phasecong(image, **settings)
    if np.isfinite(M).all() and np.isfinite(m).all():
        return M, m

    return _repair_zero_amplitude_moments(image, pc, responses, norient)


def _repair_zero_amplitude_moments(image, pc, responses, norient):
    """Repair only Phasepack 1.5's zero-amplitude Energy / sumAn divisions.

    Retain the library's covariance normalization and 1e-4 epsilon, without
    hiding unexplained numerical failures behind recomputed moment images.
    """
    repaired = False
    covx2 = np.zeros_like(image)
    covy2 = np.zeros_like(image)
    covxy = np.zeros_like(image)
    for orientation, values in enumerate(pc):
        invalid = ~np.isfinite(values)
        if invalid.any():
            amplitude = sum(np.abs(response) for response in responses[orientation])
            if np.any(amplitude[invalid] != 0):
                raise FloatingPointError("Phasepack returned non-finite PC at nonzero energy")
            values = values.copy()
            values[invalid] = 0.0
            repaired = True
        angle = orientation * np.pi / norient
        x, y = values * np.cos(angle), values * np.sin(angle)
        covx2 += x * x
        covy2 += y * y
        covxy += x * y
    if not repaired:
        raise FloatingPointError("Phasepack returned non-finite moments without zero-amplitude PC values")
    covx2 /= norient / 2.0
    covy2 /= norient / 2.0
    covxy *= 4.0 / norient
    difference = np.hypot(covxy, covx2 - covy2) + 1e-4
    M = (covx2 + covy2 + difference) / 2.0
    m = (covx2 + covy2 - difference) / 2.0
    if not (np.isfinite(M).all() and np.isfinite(m).all()):
        raise FloatingPointError("Phasepack returned non-finite moment images")
    return M, m


def _summarize_moments(M, m, mask, ddof):
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != M.shape or mask.dtype.kind not in "buif" or not np.isfinite(mask).all():
            raise ValueError("mask must be a finite numeric/boolean array matching image.shape")
        mask = mask != 0  # accepts bool, 0/1, and 0/255 masks
        M, m = M[mask], m[mask]
    if isinstance(ddof, (bool, np.bool_)) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a nonnegative integer")
    if M.size <= ddof:
        raise ValueError("statistics require more than ddof selected pixels; mask may be empty")
    return np.array([M.mean(), M.std(ddof=ddof), m.mean(), m.std(ddof=ddof)])


def phase_congruency_features(image, mask=None, *, ddof=0, **parameters):
    """Return [mean(M), std(M), mean(m), std(m)] in pc_feature_names order.

    An optional mask selects pixels AFTER filtering; it does not zero the
    grayscale input or count excluded pixels as zeros. Use a supplied ROI if
    background/padding should be excluded. No mask is inferred automatically.
    ddof=1 selects sample std; the paper does not specify this convention.
    """
    M, m = phase_congruency_moments(image, **parameters)
    return _summarize_moments(M, m, mask, ddof)


def extract_features(
    dataset_dir, class_subfolders, features_dir, save_images=False, *,
    resize=(256, 256), ddof=0, phasecong_parameters=None,
):
    """Extract one row per image and export Phase_congruency_features.csv.

    Uses the other extractors' four positional arguments and a trailing Class
    column. Returns the DataFrame as well. Rows follow class_subfolders order,
    then sorted filenames; align other feature tables by image before merging.
    Unreadable images raise an error rather than silently dropping a row.
    Statistics use the entire processed image, including any resize padding.
    For explicit ROI statistics, use phase_congruency_features(image, mask).
    save_images writes M/m PNG previews beneath features_dir/phase_congruency;
    preview clipping does not affect the floating-point descriptors.
    phasecong_parameters overrides the four-scale / six-orientation defaults;
    pass {"nscale": 5} to retain the previous scale count.
    """
    import cv2
    import pandas as pd

    settings = _validate_phasecong_parameters(
        {} if phasecong_parameters is None else phasecong_parameters
    )
    if resize is not None:
        from ..preprocessing import resize_with_padding

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    rows = []
    output_dir = Path(features_dir)
    for label in class_subfolders:
        class_dir = Path(dataset_dir) / label
        paths = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)
        if not paths:
            raise ValueError(f"No supported images found in {class_dir}")
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {path}")
            try:
                if resize is not None:
                    image = resize_with_padding(image, resize)
                print(f"Computing pc for {path}")
                M, m = phase_congruency_moments(image, **settings)
                rows.append([*_summarize_moments(M, m, None, ddof), label])
            except (ValueError, FloatingPointError) as error:
                raise type(error)(f"Failed to extract phase congruency from {path}: {error}") from error
            if save_images:
                preview_dir = output_dir / "phase_congruency" / label
                preview_dir.mkdir(parents=True, exist_ok=True)
                # Windows filenames are case-insensitive: M/m would collide.
                for name, moment in (("maximum", M), ("minimum", m)):
                    preview_path = preview_dir / f"{path.name}_{name}.png"
                    preview = np.rint(np.clip(moment, 0, 1) * 255).astype(np.uint8)
                    if not cv2.imwrite(str(preview_path), preview):
                        raise OSError(f"Failed to save moment preview: {preview_path}")

    if not rows:
        raise ValueError("No images selected for phase-congruency extraction")
    frame = pd.DataFrame(rows, columns=[*pc_feature_names, "Class"])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "Phase_congruency_features.csv", index=False)
    return frame
