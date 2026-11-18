"""Fixed Canny silhouette candidates used only by geometry diagnostics."""

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from microalgae.imaging.preprocessing.configured import PreparedInput


CANNY_MASK_PROFILES = {
    "canny_50_100_mask": [50, 100],
    "canny_100_150_mask": [100, 150],
    "canny_150_175_mask": [150, 175],
}
CANNY_REFINEMENT_PROFILES = {
    "canny_20_40_mask": {"thresholds": [20, 40], "closing_shape": "ellipse"},
    "canny_30_60_mask": {"thresholds": [30, 60], "closing_shape": "ellipse"},
    "canny_50_100_rect_mask": {"thresholds": [50, 100], "closing_shape": "rectangle"},
    "canny_20_40_rect_mask": {"thresholds": [20, 40], "closing_shape": "rectangle"},
}
ALL_CANNY_PROFILES = (*CANNY_MASK_PROFILES, *CANNY_REFINEMENT_PROFILES)
CANNY_MASK_OPERATIONS = {
    "gaussian_kernel": [3, 3], "gaussian_sigma": 0,
    "aperture_size": 3, "L2gradient": False,
    "closing_kernel": "3x3 ellipse", "closing_iterations": 1,
    "morphology_border": "constant zero",
    "fill_connectivity": 4, "component_connectivity": 8,
    "require_newly_filled_interior": True, "min_component_area": 25,
    "order": ["content_roi", "grayscale", "gaussian", "canny", "closing",
              "fill_holes", "enclosed_component_selection", "area_filter", "restore_canvas"],
}


def enclosed_mask(closed, min_area=25):
    """Retain filled components with an enclosed interior, excluding open strokes."""
    edges = closed != 0
    filled = binary_fill_holes(edges)
    interior = filled & ~edges
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        filled.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[np.unique(labels[interior])] = True
    keep &= stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def prepare_canny_mask(canvas, content, thresholds, *, closing_shape="ellipse"):
    """Reuse the reference canvas; all filtering and edge work stays inside content."""
    if (canvas.dtype != np.uint8 or canvas.ndim not in (2, 3)
            or content.dtype != bool or content.shape != canvas.shape[:2] or not content.any()):
        raise ValueError("Canny masks require a uint8 canvas and aligned nonempty Boolean content")
    if closing_shape not in ("ellipse", "rectangle"):
        raise ValueError("Canny closing_shape must be ellipse or rectangle")
    ys, xs = np.nonzero(content)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if not content[y0:y1, x0:x1].all():
        raise ValueError("Canny content must be rectangular")
    roi = canvas[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, *thresholds, apertureSize=3, L2gradient=False)
    # A zero exterior prevents morphology from extending strokes along the ROI border.
    padded = np.pad(edges, 2)
    closed = cv2.morphologyEx(
        padded, cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_RECT if closing_shape == "rectangle" else cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)[2:-2, 2:-2]
    mask = enclosed_mask(closed)
    restored = np.zeros(content.shape, dtype=bool)
    restored[y0:y1, x0:x1] = mask
    intermediates = {}
    for name, values in (("blurred", blurred), ("edges", edges), ("closed_edges", closed)):
        stage = np.zeros(content.shape, dtype=np.uint8)
        stage[y0:y1, x0:x1] = values
        intermediates[name] = stage
    return PreparedInput(restored.astype(np.uint8) * 255, restored, intermediates)
