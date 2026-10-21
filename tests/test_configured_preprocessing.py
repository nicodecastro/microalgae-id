"""Configured preprocessing routing, validation, and legacy compatibility."""

from copy import deepcopy
import importlib

import cv2
import numpy as np
import pandas as pd
import pytest

from microalgae.imaging.dataset import failure_report_path
from microalgae.imaging.features import (
    efd_features as efd, gabor_features as gabor, geometric_features as geom,
    glcm_features as glcm, hu_moments_features as hu, lbp_features as lbp,
    phase_congruency_features as pc,
)
from microalgae.imaging.preprocessing import configured as prep
from microalgae.imaging.preprocessing.resize import resize_with_padding
from microalgae.imaging.preprocessing.sanchez_matlab import segment_image
from training.config import load_config
from training.feature_tables import FEATURE_FILENAMES


EXTRACTORS = {"efd": efd, "gabor": gabor, "geom": geom, "glcm": glcm, "hu": hu, "lbp": lbp, "pc": pc}


def sample_image():
    y, x = np.indices((72, 96))
    image = (200 + (x + y) % 30).astype(np.uint8)
    foreground = np.zeros_like(image)
    cv2.ellipse(foreground, (48, 36), (30, 20), 17, 0, 360, 255, -1)
    image[foreground != 0] = (20 + (3 * x + 2 * y) % 60)[foreground != 0]
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def manual_input(image, settings, family):
    """Independent stage sequence using OpenCV, not prepare_feature_input."""
    resize = settings["resize"]
    if resize["enabled"]:
        pad = 0 if resize["padding_color"] == "black" else image.mean(axis=(0, 1))
        image = resize_with_padding(image, tuple(resize["target_size"]), pad_color=pad)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if settings["filter"]["enabled"]:
        image = cv2.bilateralFilter(image, *settings["filter"]["bilateral"]["params"])
    mask = None
    if family in ("efd", "gabor"):
        mask, _, _ = segment_image(image, settings["mask"]["dilation_size"], resize=None)
    if settings["threshold"]["enabled"]:
        params = settings["threshold"]["adaptive"]
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, params["block_size"], params["c"])
    if settings["segment"]["enabled"]:
        edge = settings["segment"]
        if edge["method"] == "canny":
            image = cv2.Canny(image, *edge["canny"]["thresholds"])
        else:
            x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=edge["sobel"]["ksize"])
            y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=edge["sobel"]["ksize"])
            image = cv2.convertScaleAbs(cv2.magnitude(x, y))
    if family == "gabor":
        image = image * mask.astype(np.uint8)
    elif family == "efd":
        image = mask.astype(np.uint8) * 255
    return image, mask


def test_reference_inspired_defaults_and_override_isolation():
    config = load_config()
    original = deepcopy(config)
    for family in EXTRACTORS:
        settings = prep.resolve_preprocessing(config, family)
        assert settings["resize"] == {"enabled": True, "target_size": [256, 256], "padding_color": "black"}
        assert settings["filter"]["enabled"] == (family == "hu")
        assert settings["threshold"]["enabled"] == (family == "hu")
        assert settings["segment"]["enabled"] == (family == "geom")
    config["preprocessing"]["overrides"]["lbp"] = {"segment": {"enabled": True, "method": "sobel"}}
    assert prep.resolve_preprocessing(config, "lbp")["segment"]["method"] == "sobel"
    assert not prep.resolve_preprocessing(config, "glcm")["segment"]["enabled"]
    config["preprocessing"]["overrides"].pop("lbp")
    assert config == original


def test_grayscale_cannot_be_silently_overridden():
    config = load_config()
    config["preprocessing"]["grayscale"] = False
    with pytest.raises(ValueError, match="grayscale"):
        prep.resolve_preprocessing(config, "lbp")
    config["dataset"]["grayscale"] = False
    config["preprocessing"]["overrides"]["lbp"] = {"grayscale": True}
    with pytest.raises(ValueError, match="grayscale"):
        prep.resolve_preprocessing(config, "lbp")


@pytest.mark.parametrize("family", EXTRACTORS)
def test_prepared_inputs_match_manual_stages(family):
    settings = prep.resolve_preprocessing(load_config(), family)
    image = sample_image()
    prepared = prep.prepare_feature_input(image, settings, family)
    expected, mask = manual_input(image, settings, family)
    np.testing.assert_array_equal(prepared.image, expected)
    if mask is None:
        assert prepared.mask is None
    else:
        assert prepared.mask.dtype == bool
        assert prepared.mask.shape == prepared.image.shape
        np.testing.assert_array_equal(prepared.mask, mask)


@pytest.mark.parametrize("family", EXTRACTORS)
def test_configured_exports_match_descriptor_calls_and_exact_previews(tmp_path, family):
    root = tmp_path / "raw"
    (root / "a").mkdir(parents=True)
    path = root / "a" / "image.png"
    assert cv2.imwrite(str(path), sample_image())
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    settings = prep.resolve_preprocessing(load_config(), family)
    prepared_image, mask = manual_input(image, settings, family)
    module = EXTRACTORS[family]
    options = {"standardize": False} if family in ("geom", "hu", "lbp", "glcm") else {}
    frame = module.extract_features(root, ["a"], tmp_path / "features", True, preprocessing=settings, **options)
    calculations = {
        "efd": lambda: efd.extract_efd_features(mask, "image.png", "a"),
        "gabor": lambda: gabor.extract_gabor_statistics(prepared_image, mask),
        "geom": lambda: geom.extract_geometric_features(prepared_image),
        "glcm": lambda: glcm.extract_glcm_statistics(prepared_image),
        "hu": lambda: hu.extract_hu_moments_from_binary(prepared_image),
        "lbp": lambda: lbp.extract_lbp_histogram(prepared_image),
        "pc": lambda: pc.phase_congruency_features(prepared_image),
    }
    assert frame.image_id.tolist() == ["a/image.png"]
    assert frame.Class.tolist() == ["a"]
    np.testing.assert_allclose(frame.iloc[0, :-2].to_numpy(dtype=float), calculations[family](), rtol=1e-12, atol=1e-12)
    preview_dir = tmp_path / "features" / "previews" / family / "a"
    np.testing.assert_array_equal(cv2.imread(str(preview_dir / "image.png_image.png"), 0), prepared_image)
    if mask is not None:
        np.testing.assert_array_equal(cv2.imread(str(preview_dir / "image.png_mask.png"), 0), mask.astype(np.uint8) * 255)
    assert pd.read_csv(failure_report_path(tmp_path / "features" / FEATURE_FILENAMES[family])).empty
    # Configured previews belong to this run, never the legacy data/success tree.
    assert not (tmp_path / "data").exists()


def test_each_stage_runs_once_and_mask_branches_before_edges(monkeypatch):
    calls = []
    captured = {}
    for name in ["resize_with_padding", "to_grayscale", "bilateral_filter", "adaptive_threshold", "sobel"]:
        original = getattr(prep, name)

        def stage(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            result = _original(*args, **kwargs)
            captured[_name] = result.copy()
            return result

        monkeypatch.setattr(prep, name, stage)

    def segmentation(image, dilation_size, *, resize):
        calls.append("mask")
        assert resize is None and dilation_size == 2
        np.testing.assert_array_equal(image, captured["bilateral_filter"])
        mask = np.zeros_like(image, dtype=bool)
        mask[40:200, 40:200] = True
        return mask, image, mask

    monkeypatch.setattr(prep, "segment_image", segmentation)
    settings = prep.validate_preprocessing({
        "filter": {"enabled": True}, "threshold": {"enabled": True},
        "segment": {"enabled": True, "method": "sobel", "sobel": {"ksize": 5}},
        "mask": {"dilation_size": 2},
    }, "gabor")
    result = prep.prepare_feature_input(sample_image(), settings, "gabor")
    assert calls == ["resize_with_padding", "to_grayscale", "bilateral_filter", "mask", "adaptive_threshold", "sobel"]
    np.testing.assert_array_equal(result.image, captured["sobel"] * result.mask)


def test_disabled_stages_do_not_run(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled stage ran")

    for name in ["resize_with_padding", "bilateral_filter", "adaptive_threshold", "canny", "sobel", "segment_image"]:
        monkeypatch.setattr(prep, name, forbidden)
    settings = prep.validate_preprocessing({"resize": {"enabled": False}}, "pc")
    image = sample_image()
    result = prep.prepare_feature_input(image, settings, "pc")
    np.testing.assert_array_equal(result.image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    assert result.mask is None


@pytest.mark.parametrize("color", ["black", "mean"])
def test_resize_preserves_proportions_and_padding_color(color):
    image = np.full((10, 20), 100, dtype=np.uint8)
    settings = prep.validate_preprocessing({"resize": {"target_size": [20, 20], "padding_color": color}}, "lbp")
    result = prep.prepare_feature_input(image, settings, "lbp").image
    assert result.shape == (20, 20)
    np.testing.assert_array_equal(result[5:15], np.full((10, 20), 100))
    assert np.all(result[:5] == (0 if color == "black" else 100))
    assert np.all(result[15:] == (0 if color == "black" else 100))


def test_sanchez_resize_none_avoids_hidden_resize_and_legacy_default_remains(monkeypatch):
    module = importlib.import_module("microalgae.imaging.preprocessing.sanchez_matlab")
    original = module.resize_with_padding
    calls = []

    def resize(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "resize_with_padding", resize)
    image = cv2.cvtColor(sample_image(), cv2.COLOR_BGR2GRAY)
    assert module.segment_image(image, resize=None)[0].shape == image.shape
    assert calls == []
    assert module.segment_image(image)[0].shape == (256, 256)
    assert calls == [1]


@pytest.mark.parametrize("settings,family", [
    ({"unknown": 1}, "lbp"), ({"grayscale": False}, "lbp"),
    ({"resize": {"enabled": 1}}, "pc"), ({"resize": {"target_size": [0, 20]}}, "lbp"),
    ({"resize": {"padding_color": "white"}}, "lbp"),
    ({"resize": {"target_size": [512, 512]}}, "gabor"),
    ({"resize": {"enabled": False}}, "gabor"),
    ({"filter": {"enabled": True, "method": "typo"}}, "hu"),
    ({"filter": {"enabled": True, "bilateral": {"params": [2, np.inf, 3]}}}, "hu"),
    ({"threshold": {"enabled": True, "adaptive": {"block_size": 4}}}, "hu"),
    ({"threshold": {"enabled": True, "adaptive": {"c": np.nan}}}, "hu"),
    ({"segment": {"enabled": True, "method": "invalid"}}, "geom"),
    ({"segment": {"enabled": True, "canny": {"thresholds": [20, 10]}}}, "geom"),
    ({"segment": {"enabled": True, "method": "sobel", "sobel": {"ksize": 2}}}, "lbp"),
    ({"threshold": {"enabled": True}}, "efd"), ({"segment": {"enabled": True}}, "efd"),
    ({"mask": {"dilation_size": 0}}, "gabor"), ({"mask": {"method": "other"}}, "efd"),
])
def test_invalid_settings_rejected(settings, family):
    with pytest.raises(ValueError):
        prep.validate_preprocessing(settings, family)


@pytest.mark.parametrize("family", EXTRACTORS)
def test_each_extractor_validates_before_writing(tmp_path, family):
    with pytest.raises(ValueError, match="grayscale"):
        EXTRACTORS[family].extract_features(tmp_path / "missing", ["a"], tmp_path / "output", preprocessing={"grayscale": False})
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("family", EXTRACTORS)
def test_preview_failure_uses_existing_image_ledger(tmp_path, monkeypatch, family):
    root = tmp_path / "raw"
    (root / "a").mkdir(parents=True)
    assert cv2.imwrite(str(root / "a" / "image.png"), sample_image())
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError, match="Failed to save prepared input"):
        EXTRACTORS[family].extract_features(root, ["a"], tmp_path / "output", True, preprocessing=prep.resolve_preprocessing(load_config(), family))
    output = tmp_path / "output" / FEATURE_FILENAMES[family]
    assert not output.exists()
    ledger = pd.read_csv(failure_report_path(output))
    assert ledger.image_id.tolist() == ["a/image.png"]
    assert ledger.error_type.tolist() == ["OSError"]


@pytest.mark.parametrize("family", EXTRACTORS)
def test_preprocessing_failure_uses_existing_image_ledger(tmp_path, monkeypatch, family):
    root = tmp_path / "raw"
    (root / "a").mkdir(parents=True)
    assert cv2.imwrite(str(root / "a" / "image.png"), sample_image())

    def fail(*args, **kwargs):
        raise ValueError("preprocessing failed")

    # PC imports the helper locally; the other modules keep a module binding.
    monkeypatch.setattr(prep if family == "pc" else EXTRACTORS[family], "prepare_feature_input", fail)
    with pytest.raises(ValueError, match="preprocessing failed"):
        EXTRACTORS[family].extract_features(root, ["a"], tmp_path / "output", preprocessing={})
    ledger = pd.read_csv(failure_report_path(tmp_path / "output" / FEATURE_FILENAMES[family]))
    assert ledger.image_id.tolist() == ["a/image.png"]
    assert ledger.error_type.tolist() == ["ValueError"]
    assert "preprocessing failed" in ledger.reason.iloc[0]


@pytest.mark.parametrize("family", ["efd", "gabor"])
def test_empty_masks_are_reported(tmp_path, monkeypatch, family):
    root = tmp_path / "raw"
    (root / "a").mkdir(parents=True)
    assert cv2.imwrite(str(root / "a" / "image.png"), sample_image())
    monkeypatch.setattr(prep, "segment_image", lambda image, **kwargs: (np.zeros_like(image, dtype=bool), image, image))
    with pytest.raises(ValueError, match="empty.*mask"):
        EXTRACTORS[family].extract_features(root, ["a"], tmp_path / "output", preprocessing={})
    ledger = pd.read_csv(failure_report_path(tmp_path / "output" / FEATURE_FILENAMES[family]))
    assert ledger.image_id.tolist() == ["a/image.png"]


def test_configured_variants_and_pc_resize_precedence(tmp_path):
    root = tmp_path / "raw"
    (root / "a").mkdir(parents=True)
    image = sample_image()
    assert cv2.imwrite(str(root / "a" / "image.png"), image)
    settings = prep.validate_preprocessing({"resize": {"target_size": [64, 48]}}, "glcm")
    expected, _ = manual_input(image, settings, "glcm")
    frame = glcm.extract_features(root, ["a"], tmp_path / "output", preprocessing=settings, mode="orientation_pooled", standardize=False)
    np.testing.assert_array_equal(frame.iloc[0, :-2].to_numpy(dtype=float), glcm.extract_glcm_statistics(expected, mode="orientation_pooled"))
    frame = pc.extract_features(root, ["a"], tmp_path / "output", preprocessing=settings, resize=(200, 200))
    np.testing.assert_allclose(frame.iloc[0, :-2].to_numpy(dtype=float), pc.phase_congruency_features(expected))
    frame = efd.extract_features(root, ["a"], tmp_path / "output", preprocessing=settings, normalize_shape=True)
    _, mask = manual_input(image, settings, "efd")
    np.testing.assert_array_equal(frame.iloc[0, :-2].to_numpy(dtype=float), efd.extract_efd_features(mask, "image.png", "a", normalize_shape=True))
