"""Extraction-only runner and command on temporary, keyed datasets."""

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from microalgae.imaging.dataset import failure_report_path
from training.config import load_config
from training.extraction import extract_configured_features
from training.feature_tables import FEATURE_FILENAMES


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "raw"
    y, x = np.indices((72, 96))
    image = (200 + (x + y) % 30).astype(np.uint8)
    mask = np.zeros_like(image)
    cv2.ellipse(mask, (48, 36), (30, 20), 17, 0, 360, 255, -1)
    image[mask != 0] = (20 + (3 * x + 2 * y) % 60)[mask != 0]
    for label in ["b", "a"]:
        (root / label).mkdir(parents=True)
        assert cv2.imwrite(str(root / label / "image.png"), image)
    config = load_config()
    config["dataset"]["raw_dir"] = str(root)
    config["dataset"]["classes"] = ["b", "a"]
    config["dataset"]["features_dir"] = str(tmp_path / "existing_features")
    config["dataset"]["combined_features_dir"] = str(tmp_path / "combined")
    config["logging"]["save_intermediate_images"] = True
    return config


def test_runner_uses_configured_order_and_returns_frames_without_training(config, tmp_path):
    config["features"]["feature_list"] = ["pc", "lbp", "hu"]
    original = deepcopy(config)
    frames = extract_configured_features(config, tmp_path / "run")
    assert list(frames) == ["pc", "lbp", "hu"]
    for family, frame in frames.items():
        assert frame.image_id.tolist() == ["b/image.png", "a/image.png"]
        assert frame.Class.tolist() == ["b", "a"]
        assert frame.columns[-2:].tolist() == ["image_id", "Class"]
        assert (tmp_path / "run" / FEATURE_FILENAMES[family]).is_file()
    assert not (tmp_path / "run" / FEATURE_FILENAMES["glcm"]).exists()
    assert pd.read_csv(tmp_path / "run" / "feature_table_audit.csv").empty
    assert not Path(config["dataset"]["features_dir"]).exists()
    assert not Path(config["dataset"]["combined_features_dir"]).exists()
    assert config == original
    snapshot = yaml.safe_load((tmp_path / "run" / "extraction_settings.yaml").read_text())
    assert list(snapshot["resolved_preprocessing"]) == ["pc", "lbp", "hu"]
    assert snapshot["configuration"] == config
    assert snapshot["resolved_preprocessing"]["hu"]["filter"]["bilateral"]["params"] == [40, 1, 25]


def test_command_exports_all_families_and_preserves_source_files(config, tmp_path):
    original_paths = list(Path(config["dataset"]["raw_dir"]).rglob("*.png"))
    original_bytes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in original_paths}
    existing_features = Path(config["dataset"]["features_dir"])
    existing_features.mkdir()
    legacy = existing_features / "LBP_features.csv"
    legacy.write_text("legacy,data\n1,2\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "new_run"
    command = [sys.executable, str(REPOSITORY_ROOT / "scripts" / "extract_features.py"),
               "--config", str(config_path), "--output-dir", str(output)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Extraction and audit passed" in result.stdout
    for family, filename in FEATURE_FILENAMES.items():
        frame = pd.read_csv(output / filename)
        assert frame.image_id.tolist() == ["b/image.png", "a/image.png"]
        assert frame.Class.tolist() == ["b", "a"]
        assert np.isfinite(frame.iloc[:, :-2].to_numpy()).all()
        assert pd.read_csv(failure_report_path(output / filename)).empty
        assert (output / "previews" / family / "a" / "image.png_image.png").is_file()
    assert (output / "previews" / "gabor" / "a" / "image.png_mask.png").is_file()
    assert (output / "previews" / "efd" / "a" / "image.png_mask.png").is_file()
    assert not (output / "previews" / "pc" / "a" / "image.png_mask.png").exists()
    assert pd.read_csv(output / "feature_table_audit.csv").empty
    manifest = pd.read_csv(output / "dataset_manifest.csv")
    assert manifest.image_id.tolist() == ["b/image.png", "a/image.png"]
    assert manifest.parent_image_id.isna().all()
    assert legacy.read_text() == "legacy,data\n1,2\n"
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in original_paths} == original_bytes
    assert not Path(config["dataset"]["combined_features_dir"]).exists()
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("problem", ["grayscale", "gabor_size", "efd_edges", "unknown_family", "duplicate_family", "bad_stage", "unknown_override"])
def test_preflight_failures_write_nothing(config, tmp_path, problem):
    if problem == "grayscale":
        config["dataset"]["grayscale"] = False
    elif problem == "gabor_size":
        config["preprocessing"]["resize"]["target_size"] = [512, 512]
    elif problem == "efd_edges":
        config["preprocessing"]["segment"]["enabled"] = True
    elif problem == "unknown_family":
        config["features"]["feature_list"] = ["wrong"]
    elif problem == "duplicate_family":
        config["features"]["feature_list"] = ["lbp", "lbp"]
    elif problem == "bad_stage":
        config["preprocessing"]["overrides"]["hu"]["filter"]["method"] = "wrong"
    else:
        config["preprocessing"]["overrides"]["wrong"] = {}
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        extract_configured_features(config, output)
    assert not output.exists()


def test_existing_output_and_raw_directory_targets_rejected(config, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    original = output / "keep.txt"
    original.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be new"):
        extract_configured_features(config, output)
    assert original.read_text() == "keep"
    raw_output = Path(config["dataset"]["raw_dir"]) / "run"
    with pytest.raises(ValueError, match="raw dataset"):
        extract_configured_features(config, raw_output)
    assert not raw_output.exists()


def test_geometry_skips_fail_audit_but_do_not_fabricate_rows(config, tmp_path):
    config["features"]["feature_list"] = ["geom", "lbp"]
    path = Path(config["dataset"]["raw_dir"]) / "a" / "blank.png"
    assert cv2.imwrite(str(path), np.zeros((72, 96), dtype=np.uint8))
    output = tmp_path / "run"
    with pytest.warns(UserWarning, match="Skipping"), pytest.raises(ValueError, match="audit failed"):
        extract_configured_features(config, output)
    geom = pd.read_csv(output / FEATURE_FILENAMES["geom"])
    assert "a/blank.png" not in geom.image_id.tolist()
    assert "a/blank.png" in pd.read_csv(output / FEATURE_FILENAMES["lbp"]).image_id.tolist()
    issues = pd.read_csv(output / "feature_table_audit.csv")
    assert {"missing_image", "extraction_failure"} <= set(issues.issue)


def test_unreadable_image_keeps_error_report_and_audit(config, tmp_path):
    config["features"]["feature_list"] = ["lbp", "pc"]
    (Path(config["dataset"]["raw_dir"]) / "b" / "broken.png").touch()
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="Failed to load image"):
        extract_configured_features(config, output)
    ledger = pd.read_csv(failure_report_path(output / FEATURE_FILENAMES["lbp"]))
    assert ledger.image_id.tolist() == ["b/broken.png"]
    assert (output / "feature_table_audit.csv").is_file()
    assert not (output / FEATURE_FILENAMES["pc"]).exists()


def test_command_returns_nonzero_for_invalid_config(config, tmp_path, capsys):
    from scripts.extract_features import main

    config["preprocessing"]["resize"]["target_size"] = [512, 512]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["--config", str(path), "--output-dir", str(tmp_path / "run")]) == 1
    assert "256 x 256" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("standardize", [True, False])
def test_standardization_flag_only_routes_to_four_supported_families(config, tmp_path, monkeypatch, standardize):
    import training.extraction as extraction

    calls = {}

    def module(name):
        family = next(family for family, suffix in extraction.MODULES.items() if name.endswith("." + suffix))

        def extract(root, classes, output, **kwargs):
            calls[family] = kwargs
            frame = pd.DataFrame({"value": [3., 7.], "image_id": ["b/image.png", "a/image.png"], "Class": classes})
            frame.to_csv(output / FEATURE_FILENAMES[family], index=False)
            return frame

        return SimpleNamespace(extract_features=extract)

    monkeypatch.setattr(extraction, "import_module", module)
    output = tmp_path / "flag_test"
    # Omitting the flag must still forward the original True default.
    kwargs = {} if standardize else {"standardize": False}
    extract_configured_features(config, output, **kwargs)
    for family, options in calls.items():
        if family in {"geom", "hu", "lbp", "glcm"}:
            assert options["standardize"] is standardize
        else:
            assert "standardize" not in options
    assert yaml.safe_load((output / "extraction_settings.yaml").read_text())["standardize"] is standardize


def test_invalid_standardization_flag_writes_nothing(config, tmp_path):
    with pytest.raises(ValueError, match="standardize must be Boolean"):
        extract_configured_features(config, tmp_path / "invalid", standardize="False")
    assert not (tmp_path / "invalid").exists()
