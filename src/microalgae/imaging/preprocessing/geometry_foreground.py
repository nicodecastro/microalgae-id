"""Experimental component masks for geometry, never inferred as ground truth."""

from numbers import Integral, Real

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_propagation

from .adaptive import adaptive_threshold
from .bilateral import bilateral_filter

CLAHE_PARAMETERS = {
    "otsu_background_clahe": (1.5, (4, 4)),
    "otsu_background_clahe_clip1": (1.0, (4, 4)),
    "otsu_background_clahe_grid2": (1.5, (2, 2)),
}
FOREGROUND_METHODS = ("otsu", "otsu_bright", "adaptive", "otsu_background",
                      "otsu_background_hysteresis", *CLAHE_PARAMETERS)


def validate_hysteresis(low_ratio, clahe_params=None):
    if (isinstance(low_ratio, bool) or not isinstance(low_ratio, Real)
            or not np.isfinite(low_ratio) or not 0 < low_ratio <= 1):
        raise ValueError("foreground.hysteresis.low_ratio must be finite and in (0, 1]")
    if clahe_params is not None:
        if not isinstance(clahe_params, dict) or set(clahe_params) != {"clip_limit", "tile_grid_size"}:
            raise ValueError("hysteresis CLAHE requires clip_limit and tile_grid_size")
        clip, grid = clahe_params["clip_limit"], clahe_params["tile_grid_size"]
        if isinstance(clip, bool) or not isinstance(clip, Real) or not np.isfinite(clip) or clip <= 0:
            raise ValueError("hysteresis CLAHE clip_limit must be positive and finite")
        if (not isinstance(grid, (list, tuple)) or len(grid) != 2
                or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in grid)):
            raise ValueError("hysteresis CLAHE tile_grid_size requires two positive integers")


def geometry_foreground_mask(
    gray, content_mask, *, method="otsu", min_component_area=25,
    block_size=31, c=3, bilateral_params=None, background_kernel_size=61,
    intermediates=None, hysteresis_low_ratio=0.5, hysteresis_clahe_params=None,
):
    """Threshold real image content, preserving all sufficiently large components.

    Area filtering uses foreground pixel counts, not contour area. Padding is
    excluded from denoising, threshold estimation, and morphology. Components
    touching the content boundary are retained; diagnostics flag them separately.
    ``otsu`` selects dark foreground; ``otsu_bright`` selects bright foreground.
    ``otsu_background`` thresholds closing-minus-image contrast for dark objects.
    ``otsu_background_clahe`` first applies fixed exploratory CLAHE (clip 1.5,
    4x4 tile grid), after optional bilateral filtering and before Gaussian blur.
    Its ``_clip1`` and ``_grid2`` variants change only clip limit to 1.0 or
    tile grid to 2x2 respectively; extraction defaults remain unchanged.
    """
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
        raise ValueError("Geometry foreground requires a nonempty 2-D uint8 image")
    content_mask = np.asarray(content_mask)
    if content_mask.dtype != bool or content_mask.shape != gray.shape or not content_mask.any():
        raise ValueError("content_mask must be a nonempty Boolean region matching the image")
    if method not in FOREGROUND_METHODS:
        raise ValueError(f"Unknown foreground.method; expected one of {FOREGROUND_METHODS}")
    if method == "otsu_background_hysteresis":
        validate_hysteresis(hysteresis_low_ratio, hysteresis_clahe_params)
    elif hysteresis_clahe_params is not None:
        raise ValueError("Hysteresis CLAHE requires method otsu_background_hysteresis")
    background_method = method in ("otsu_background", "otsu_background_hysteresis") or method in CLAHE_PARAMETERS
    if background_method and (
        isinstance(background_kernel_size, bool) or not isinstance(background_kernel_size, Integral)
        or background_kernel_size < 3 or background_kernel_size % 2 == 0
    ):
        raise ValueError("foreground.background.kernel_size must be an odd integer >= 3")
    if isinstance(min_component_area, bool) or not isinstance(min_component_area, Integral) or min_component_area < 1:
        raise ValueError("foreground.min_component_area must be a positive integer")
    if method == "adaptive" and (
        isinstance(block_size, bool) or not isinstance(block_size, Integral)
        or block_size < 3 or block_size % 2 == 0
        or isinstance(c, bool) or not isinstance(c, Real) or not np.isfinite(c)
    ):
        raise ValueError("foreground.adaptive requires an odd block_size >= 3 and finite c")
    ys, xs = np.nonzero(content_mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if not content_mask[y0:y1, x0:x1].all():
        raise ValueError("content_mask must describe a rectangular image-content region")
    roi = gray[y0:y1, x0:x1]
    result = np.zeros_like(content_mask)
    if np.ptp(roi) == 0 and not background_method:
        return result
    if bilateral_params is not None:
        roi = bilateral_filter(roi, *bilateral_params)
        if intermediates is not None:
            stage = np.zeros_like(gray)
            stage[y0:y1, x0:x1] = roi
            intermediates["denoised"] = stage
    if method in CLAHE_PARAMETERS or hysteresis_clahe_params is not None:
        clip_limit, grid_size = (CLAHE_PARAMETERS[method] if method in CLAHE_PARAMETERS else
                                (hysteresis_clahe_params["clip_limit"], hysteresis_clahe_params["tile_grid_size"]))
        roi = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size).apply(roi)
        if intermediates is not None:
            stage = np.zeros_like(gray)
            stage[y0:y1, x0:x1] = roi
            intermediates["contrast"] = stage
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    if background_method:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (background_kernel_size, background_kernel_size))
        background = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        residual = cv2.subtract(background, roi)
        if intermediates is not None:
            for name, values in (("background", background), ("residual", residual)):
                stage = np.zeros_like(gray)
                stage[y0:y1, x0:x1] = values
                intermediates[name] = stage
        roi = residual
    if np.ptp(roi) == 0:
        return result
    if method in ("otsu", "otsu_bright") or background_method:
        polarity = cv2.THRESH_BINARY_INV if method == "otsu" else cv2.THRESH_BINARY
        high, binary = cv2.threshold(roi, 0, 255, polarity | cv2.THRESH_OTSU)
        if method == "otsu_background_hysteresis":
            seeds, allowed = binary != 0, roi > hysteresis_low_ratio * high
            if intermediates is not None:
                for name, values in (("seeds", seeds), ("growth_allowed", allowed)):
                    stage = np.zeros_like(gray)
                    stage[y0:y1, x0:x1] = values.astype(np.uint8) * 255
                    intermediates[name] = stage
            binary = binary_propagation(seeds, structure=np.ones((3, 3), bool), mask=allowed).astype(np.uint8) * 255
    else:
        binary = adaptive_threshold(roi, int(block_size), float(c))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    binary = binary_fill_holes(binary != 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_component_area
    keep[0] = False
    result[y0:y1, x0:x1] = keep[labels]
    return result
