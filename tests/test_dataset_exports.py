"""All seven dataset writers share identity and failure-ledger contracts."""

import cv2
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from microalgae.imaging.dataset import build_dataset_manifest, failure_report_path
from microalgae.imaging.features import (
    efd_features as efd, gabor_features as gabor, geometric_features as geom,
    glcm_features as glcm, hu_moments_features as hu, lbp_features as lbp,
    phase_congruency_features as pc,
)
from training.feature_tables import FEATURE_FILENAMES, audit_feature_tables


EXTRACTORS = {"efd": efd, "gabor": gabor, "geom": geom, "glcm": glcm, "hu": hu, "lbp": lbp, "pc": pc}


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "raw"
    image = np.full((72, 72), 255, dtype=np.uint8)
    cv2.ellipse(image, (36, 36), (24, 13), 17, 0, 360, 0, -1)
    for label in ["b", "a"]:
        (root / label).mkdir(parents=True)
        assert cv2.imwrite(str(root / label / "shared.PNG"), image)
    # Existing regression tests cover real segmentation and Gabor math. Keep
    # the cross-writer contract tests independent of filter-bank construction.
    monkeypatch.setattr(gabor, "_LogGaborBank", lambda: object())
    monkeypatch.setattr(gabor, "extract_gabor_statistics", lambda *args, **kwargs: np.arange(964, dtype=float))
    for module in [efd, gabor]:
        monkeypatch.setattr(
            module, "segment_image",
            lambda image, dilation_size: ((image == 0).astype(np.uint8) * 255, image, image),
        )
    return root


@pytest.mark.parametrize("family", EXTRACTORS)
def test_every_export_has_portable_ids_and_empty_success_ledger(dataset, tmp_path, family):
    output_dir = tmp_path / "features"
    frame = EXTRACTORS[family].extract_features(dataset, ["b", "a"], output_dir)
    assert frame.columns[-2:].tolist() == ["image_id", "Class"]
    assert frame.image_id.tolist() == ["b/shared.PNG", "a/shared.PNG"]
    assert frame.Class.tolist() == ["b", "a"]
    output = output_dir / FEATURE_FILENAMES[family]
    pd.testing.assert_frame_equal(frame, pd.read_csv(output))
    assert pd.read_csv(failure_report_path(output)).empty
    assert audit_feature_tables(build_dataset_manifest(dataset, ["b", "a"]), {family: output}).empty


@pytest.mark.parametrize("family", EXTRACTORS)
def test_every_writer_records_unreadable_image_without_fabrication(tmp_path, family):
    root = tmp_path / "raw"
    (root / "a").mkdir(parents=True)
    (root / "a" / "broken.png").touch()
    output_dir = tmp_path / "features"
    if family == "geom":
        with pytest.warns(UserWarning, match="Skipping"), pytest.raises(ValueError, match="No valid"):
            geom.extract_features(root, ["a"], output_dir)
    else:
        with pytest.raises(ValueError, match="Failed to load image"):
            EXTRACTORS[family].extract_features(root, ["a"], output_dir)
    output = output_dir / FEATURE_FILENAMES[family]
    assert not output.exists()
    failures = pd.read_csv(failure_report_path(output))
    assert failures.columns.tolist() == ["image_id", "Class", "error_type", "reason"]
    assert failures.image_id.tolist() == ["a/broken.png"]
    assert failures.Class.tolist() == ["a"]
    assert failures.error_type.tolist() == ["ValueError"]


@pytest.mark.parametrize("family", EXTRACTORS)
def test_every_writer_records_preview_failure_and_resets_on_retry(dataset, tmp_path, monkeypatch, family):
    output_dir = tmp_path / "features"
    module = EXTRACTORS[family]
    output = output_dir / FEATURE_FILENAMES[family]
    module.extract_features(dataset, ["b", "a"], output_dir)
    old_bytes = output.read_bytes()
    original_write = cv2.imwrite
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError, match="Failed to save"):
        module.extract_features(dataset, ["b", "a"], output_dir, True)
    assert output.read_bytes() == old_bytes
    failures = pd.read_csv(failure_report_path(output))
    assert failures.image_id.tolist() == ["b/shared.PNG"]
    assert failures.error_type.tolist() == ["OSError"]
    assert "extraction_failure" in audit_feature_tables(
        build_dataset_manifest(dataset, ["b", "a"]), {family: output},
    ).issue.tolist()
    monkeypatch.setattr(cv2, "imwrite", original_write)
    module.extract_features(dataset, ["b", "a"], output_dir)
    assert pd.read_csv(failure_report_path(output)).empty


def test_geometry_skips_are_reported_and_block_strict_alignment(dataset, tmp_path):
    assert cv2.imwrite(str(dataset / "b" / "blank.png"), np.zeros((72, 72), dtype=np.uint8))
    output_dir = tmp_path / "features"
    with pytest.warns(UserWarning, match="no valid contours"):
        frame = geom.extract_features(dataset, ["b", "a"], output_dir)
    assert "b/blank.png" not in frame.image_id.tolist()
    output = output_dir / FEATURE_FILENAMES["geom"]
    failures = pd.read_csv(failure_report_path(output))
    assert failures.image_id.tolist() == ["b/blank.png"]
    issues = audit_feature_tables(build_dataset_manifest(dataset, ["b", "a"]), {"geom": output})
    assert {"missing_image", "extraction_failure"} <= set(issues.issue)


def test_hu_raw_values_and_default_scaling_unchanged(dataset, tmp_path):
    image_path = dataset / "a" / "other.png"
    image = np.zeros((72, 72), dtype=np.uint8)
    cv2.circle(image, (36, 36), 20, 255, -1)
    assert cv2.imwrite(str(image_path), image)
    raw = hu.extract_features(dataset, ["b", "a"], tmp_path / "raw_features", standardize=False)
    expected = [hu.extract_hu_moments(dataset / identifier, "test", "test") for identifier in raw.image_id]
    np.testing.assert_array_equal(raw[hu.FEATURE_NAMES], expected)
    scaled = hu.extract_features(dataset, ["b", "a"], tmp_path / "scaled_features")
    np.testing.assert_array_equal(scaled[hu.FEATURE_NAMES], StandardScaler().fit_transform(expected))
    assert scaled.image_id.tolist() == raw.image_id.tolist()


@pytest.mark.parametrize("family,kwargs,filename", [
    ("glcm", {"mode": "orientation_pooled"}, "GLCM_orientation_pooled_features.csv"),
    ("efd", {"normalize_shape": True}, "EFD_shape_normalized_features.csv"),
])
def test_variants_keep_independent_failure_reports(dataset, tmp_path, family, kwargs, filename):
    output_dir = tmp_path / "features"
    module = EXTRACTORS[family]
    module.extract_features(dataset, ["b", "a"], output_dir)
    original_ledger = failure_report_path(output_dir / FEATURE_FILENAMES[family]).read_bytes()
    module.extract_features(dataset, ["b", "a"], output_dir, **kwargs)
    assert failure_report_path(output_dir / FEATURE_FILENAMES[family]).read_bytes() == original_ledger
    assert pd.read_csv(failure_report_path(output_dir / filename)).empty
