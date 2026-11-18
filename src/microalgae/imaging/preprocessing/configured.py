"""Validated per-family preprocessing, separate from descriptor calculations."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import cv2
import numpy as np

from .adaptive import adaptive_threshold
from .bilateral import bilateral_filter
from .canny import canny
from .grayscale import to_grayscale
from .geometry_foreground import CLAHE_PARAMETERS, FOREGROUND_METHODS, geometry_foreground_mask, validate_hysteresis
from .resize import resize_with_padding
from .sanchez_matlab import segment_image
from .sobel import sobel


FAMILIES = ("efd", "gabor", "geom", "glcm", "hu", "lbp", "pc")
DEFAULT_PREPROCESSING = {
    "grayscale": True,
    "resize": {"enabled": True, "target_size": [256, 256], "padding_color": "black"},
    "filter": {"enabled": False, "method": "bilateral", "bilateral": {"params": [40, 1, 25]}},
    "threshold": {"enabled": False, "method": "adaptive", "adaptive": {"block_size": 11, "c": 2}},
    "segment": {"enabled": False, "method": "canny", "canny": {"thresholds": [150, 175]}, "sobel": {"ksize": 3}},
    "mask": {"method": "sanchez", "dilation_size": 4},
}


def _family_defaults(family):
    defaults = deepcopy(DEFAULT_PREPROCESSING)
    if family == "geom":
        defaults["foreground"] = {
            "enabled": False, "method": "otsu", "min_component_area": 25,
            "adaptive": {"block_size": 31, "c": 3},
            "background": {"kernel_size": 61},
            "hysteresis": {"low_ratio": 0.5,
                           "clahe": {"enabled": False, "clip_limit": 1.5, "tile_grid_size": [4, 4]}},
        }
    return defaults


def _merge(defaults, updates, path="preprocessing"):
    if not isinstance(updates, Mapping):
        raise ValueError(f"{path} must be a mapping")
    result = deepcopy(defaults)
    for key, value in updates.items():
        if key not in defaults:
            raise ValueError(f"Unknown setting: {path}.{key}")
        result[key] = (
            _merge(defaults[key], value, f"{path}.{key}")
            if isinstance(defaults[key], dict) else deepcopy(value)
        )
    return result


def _integer(value, minimum=1):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value)


def validate_preprocessing(settings, family):
    """Resolve a partial family mapping and validate it before any file writes."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown feature family: {family!r}")
    settings = _merge(_family_defaults(family), settings)
    if settings["grayscale"] is not True:
        raise ValueError("Configured feature extraction requires grayscale=True")
    for name in ("resize", "filter", "threshold", "segment"):
        if not isinstance(settings[name]["enabled"], bool):
            raise ValueError(f"{name}.enabled must be a Boolean")
    resize = settings["resize"]
    if resize["enabled"]:
        size = resize["target_size"]
        if not isinstance(size, (list, tuple)) or len(size) != 2 or not all(_integer(v) for v in size):
            raise ValueError("resize.target_size must contain two positive integers (width, height)")
        resize["target_size"] = list(size)
        if resize["padding_color"] not in ("black", "mean"):
            raise ValueError("resize.padding_color must be 'black' or 'mean'")
    if family == "gabor" and (not resize["enabled"] or resize["target_size"] != [256, 256]):
        raise ValueError("Configured Log-Gabor requires enabled 256 x 256 resizing")
    denoise = settings["filter"]
    if denoise["enabled"]:
        params = denoise["bilateral"]["params"]
        if denoise["method"] != "bilateral":
            raise ValueError("filter.method must be 'bilateral'")
        if (not isinstance(params, (list, tuple)) or len(params) != 3
                or not _integer(params[0]) or not all(_finite(v) and v > 0 for v in params[1:])):
            raise ValueError("bilateral.params requires a positive integer diameter and two positive finite sigmas")
        denoise["bilateral"]["params"] = list(params)
    threshold = settings["threshold"]
    if threshold["enabled"]:
        block = threshold["adaptive"]["block_size"]
        if threshold["method"] != "adaptive":
            raise ValueError("threshold.method must be 'adaptive'")
        if not _integer(block, 3) or block % 2 == 0 or not _finite(threshold["adaptive"]["c"]):
            raise ValueError("adaptive requires an odd block_size >= 3 and finite c")
    edge = settings["segment"]
    if edge["enabled"]:
        if edge["method"] == "canny":
            thresholds = edge["canny"]["thresholds"]
            if (not isinstance(thresholds, (list, tuple)) or len(thresholds) != 2
                    or not all(_finite(v) and v >= 0 for v in thresholds) or thresholds[0] > thresholds[1]):
                raise ValueError("canny.thresholds requires two finite nonnegative values, low <= high")
            edge["canny"]["thresholds"] = list(thresholds)
        elif edge["method"] == "sobel":
            ksize = edge["sobel"]["ksize"]
            if not _integer(ksize) or ksize % 2 == 0 or ksize > 31:
                raise ValueError("sobel.ksize must be an odd integer from 1 to 31")
        else:
            raise ValueError("segment.method must be 'canny' or 'sobel'")
    if family == "efd" and (threshold["enabled"] or edge["enabled"]):
        raise ValueError("EFD consumes its separate mask; threshold/edge image settings are not supported")
    if family == "geom":
        foreground = settings["foreground"]
        if not isinstance(foreground["enabled"], bool):
            raise ValueError("foreground.enabled must be Boolean")
        if foreground["enabled"]:
            if threshold["enabled"] or edge["enabled"]:
                raise ValueError("Geometry foreground requires disabled threshold/edge stages")
            if foreground["method"] not in FOREGROUND_METHODS:
                raise ValueError(f"Unknown foreground.method; expected one of {FOREGROUND_METHODS}")
            if foreground["method"] == "otsu_background_hysteresis":
                hysteresis = foreground["hysteresis"]
                clahe = hysteresis["clahe"]
                if not isinstance(clahe["enabled"], bool):
                    raise ValueError("foreground.hysteresis.clahe.enabled must be Boolean")
                validate_hysteresis(hysteresis["low_ratio"], {k: clahe[k] for k in ("clip_limit", "tile_grid_size")})
            if foreground["method"] in ("otsu_background", "otsu_background_hysteresis") or foreground["method"] in CLAHE_PARAMETERS:
                kernel = foreground["background"]["kernel_size"]
                if not _integer(kernel, 3) or kernel % 2 == 0:
                    raise ValueError("foreground.background.kernel_size must be an odd integer >= 3")
            if not _integer(foreground["min_component_area"]):
                raise ValueError("foreground.min_component_area must be a positive integer")
            if foreground["method"] == "adaptive":
                block = foreground["adaptive"]["block_size"]
                if not _integer(block, 3) or block % 2 == 0 or not _finite(foreground["adaptive"]["c"]):
                    raise ValueError("foreground.adaptive requires an odd block_size >= 3 and finite c")
    mask = settings["mask"]
    if mask["method"] != "sanchez" or not _integer(mask["dilation_size"]):
        raise ValueError("mask requires method='sanchez' and a positive integer dilation_size")
    return settings


def resolve_preprocessing(config, family):
    """Merge shared YAML settings and a family override without mutating config."""
    shared = config.get("preprocessing", {})
    if not isinstance(shared, Mapping):
        raise ValueError("preprocessing must be a mapping")
    overrides = shared.get("overrides", {})
    if not isinstance(overrides, Mapping) or set(overrides) - set(FAMILIES):
        raise ValueError("preprocessing.overrides must map known feature families to settings")
    shared = {key: value for key, value in shared.items() if key != "overrides"}
    if config.get("dataset", {}).get("grayscale", True) is not True:
        raise ValueError("Configured feature extraction requires dataset.grayscale=True")
    shared.setdefault("grayscale", True)
    settings = _merge(_merge(_family_defaults(family), shared), overrides.get(family, {}))
    return validate_preprocessing(settings, family)


@dataclass
class PreparedInput:
    image: np.ndarray
    mask: np.ndarray | None = None
    intermediates: dict = field(default_factory=dict)


def prepare_feature_input(image, settings, family):
    """Prepare a uint8 image using already validated settings, exactly once.

    Masks branch from grayscale/denoised intensities, not threshold/edge output.
    The returned image is the exact descriptor input, including pre-masking for
    Gabor and binary conversion for EFD. Geometry may explicitly opt into a
    component mask; other families receive no inferred ROI.
    """
    if image is None or image.dtype != np.uint8 or not image.size or image.ndim not in (2, 3):
        raise ValueError("Expected a nonempty uint8 image")
    resize = settings["resize"]
    foreground = settings.get("foreground", {}) if family == "geom" else {}
    use_foreground = foreground.get("enabled", False)
    needs_content = use_foreground or family in ("efd", "gabor")
    content_mask = np.ones(image.shape[:2], dtype=bool) if needs_content else None
    if resize["enabled"]:
        pad = 0 if resize["padding_color"] == "black" else image.mean(axis=(0, 1))
        if needs_content:
            image, content_mask = resize_with_padding(
                image, tuple(resize["target_size"]), pad_color=pad, return_content_mask=True,
            )
        else:
            image = resize_with_padding(image, tuple(resize["target_size"]), pad_color=pad)
    image = to_grayscale(image)
    if use_foreground:
        intermediates = {}
        mask = geometry_foreground_mask(
            image, content_mask, method=foreground["method"],
            min_component_area=foreground["min_component_area"], **foreground["adaptive"],
            bilateral_params=settings["filter"]["bilateral"]["params"] if settings["filter"]["enabled"] else None,
            background_kernel_size=foreground["background"]["kernel_size"], intermediates=intermediates,
            hysteresis_low_ratio=foreground.get("hysteresis", {}).get("low_ratio", 0.5),
            hysteresis_clahe_params=(
                {k: foreground["hysteresis"]["clahe"][k] for k in ("clip_limit", "tile_grid_size")}
                if foreground["method"] == "otsu_background_hysteresis"
                and foreground.get("hysteresis", {}).get("clahe", {}).get("enabled", False) else None),
        )
        return PreparedInput(mask.astype(np.uint8) * 255, mask, intermediates)
    if settings["filter"]["enabled"]:
        image = bilateral_filter(image, *settings["filter"]["bilateral"]["params"])
    mask = None
    if family in ("efd", "gabor"):
        # Artificial padding must not influence Otsu or become the largest object.
        rows, columns = np.nonzero(content_mask)
        region = np.s_[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
        foreground_mask, _, _ = segment_image(
            image[region], dilation_size=settings["mask"]["dilation_size"], resize=None,
        )
        mask = np.zeros(image.shape, dtype=bool)
        if foreground_mask.shape != image[region].shape or not np.isfinite(foreground_mask).all():
            raise ValueError("Sánchez segmentation returned an empty or invalid foreground mask")
        mask[region] = foreground_mask
        if mask.shape != image.shape or not np.isfinite(mask).all() or not np.any(mask):
            raise ValueError("Sánchez segmentation returned an empty or invalid foreground mask")
        mask = mask != 0
    if settings["threshold"]["enabled"]:
        args = settings["threshold"]["adaptive"]
        image = adaptive_threshold(image, args["block_size"], args["c"])
    if settings["segment"]["enabled"]:
        edge = settings["segment"]
        image = (canny(image, *edge["canny"]["thresholds"]) if edge["method"] == "canny"
                 else sobel(image, edge["sobel"]["ksize"]))
    if family == "efd":
        image = mask.astype(np.uint8) * 255
    elif family == "gabor":
        image = image * mask.astype(np.uint8)
    return PreparedInput(image, mask)


def save_prepared_input(prepared, features_dir, family, identifier):
    """Save lossless previews of the actual calculation inputs, without reruns."""
    path = Path(features_dir) / "previews" / family / identifier
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [("image", prepared.image)]
    if prepared.mask is not None:
        images.append(("mask", prepared.mask.astype(np.uint8) * 255))
    images.extend(prepared.intermediates.items())
    for name, image in images:
        output = path.with_name(f"{path.name}_{name}.png")
        if not cv2.imwrite(str(output), image):
            raise OSError(f"Failed to save prepared input: {output}")
