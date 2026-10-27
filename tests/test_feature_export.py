"""Behavior at the shared dataset feature-export seam."""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from microalgae.imaging.dataset import failure_report_path


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "raw"
    for label in ("b", "a"):
        directory = root / label
        directory.mkdir(parents=True)
        for name in ("z.PNG", "a.tiff", "ignored.txt"):
            (directory / name).touch()
        (directory / "nested.jpg").mkdir()
    return root


def export(dataset, output, extract_image, **options):
    from microalgae.imaging.features._export import export_feature_table

    return export_feature_table(
        dataset, ["b", "a"], output, ["value"], extract_image,
        empty_message="No rows extracted", **options,
    )


@pytest.mark.parametrize("standardize", [False, True])
def test_export_preserves_identity_and_scales_only_features(dataset, tmp_path, standardize):
    values = iter([1.0, 3.0, 7.0, 9.0])
    output = tmp_path / "features" / "example.csv"
    frame = export(dataset, output, lambda path, label: [next(values)], standardize=standardize)

    assert frame.columns.tolist() == ["value", "image_id", "Class"]
    assert frame.image_id.tolist() == ["b/a.tiff", "b/z.PNG", "a/a.tiff", "a/z.PNG"]
    assert frame.Class.tolist() == ["b", "b", "a", "a"]
    expected = np.array([[1.0], [3.0], [7.0], [9.0]])
    if standardize:
        expected = StandardScaler().fit_transform(expected)
    np.testing.assert_array_equal(frame[["value"]], expected)
    pd.testing.assert_frame_equal(frame, pd.read_csv(output))
    assert pd.read_csv(failure_report_path(output)).empty


@pytest.mark.parametrize("sort_paths", [False, True])
def test_export_keeps_the_selected_filename_order(dataset, tmp_path, monkeypatch, sort_paths):
    original = os.listdir
    monkeypatch.setattr(os, "listdir", lambda path: sorted(original(path), reverse=True))
    frame = export(dataset, tmp_path / "example.csv", lambda path, label: [1], sort_paths=sort_paths)
    names = ["a.tiff", "z.PNG"] if sort_paths else ["z.PNG", "a.tiff"]
    assert frame.image_id.tolist() == [f"{label}/{name}" for label in ("b", "a") for name in names]


def test_skipped_image_is_warned_and_recorded_once(dataset, tmp_path):
    from microalgae.imaging.features._export import SkippedImage

    output = tmp_path / "example.csv"

    def extract_image(path, label):
        return SkippedImage("No valid contours") if label == "b" and path.name == "a.tiff" else [1]

    with pytest.warns(UserWarning, match="No valid contours") as warnings:
        frame = export(dataset, output, extract_image)
    assert len(warnings) == 1
    assert "b/a.tiff" not in frame.image_id.tolist()
    ledger = pd.read_csv(failure_report_path(output))
    assert ledger.to_dict("records") == [{
        "image_id": "b/a.tiff", "Class": "b", "error_type": "ValueError", "reason": "No valid contours",
    }]


def test_failure_preserves_previous_table_and_retry_resets_ledger(dataset, tmp_path):
    output = tmp_path / "example.csv"
    export(dataset, output, lambda path, label: [1])
    original = output.read_bytes()
    error = OSError("Preview write failed")

    def fail(path, label):
        raise error

    with pytest.raises(OSError, match="Preview write failed") as raised:
        export(dataset, output, fail)
    assert raised.value is error
    assert output.read_bytes() == original
    ledger = pd.read_csv(failure_report_path(output))
    assert ledger.image_id.tolist() == ["b/a.tiff"]
    assert ledger.error_type.tolist() == ["OSError"]
    export(dataset, output, lambda path, label: [2])
    assert pd.read_csv(failure_report_path(output)).empty
    assert pd.read_csv(output).value.tolist() == [2, 2, 2, 2]


def test_all_skips_leave_ledger_without_writing_table(dataset, tmp_path):
    from microalgae.imaging.features._export import SkippedImage

    output = tmp_path / "example.csv"
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="No rows extracted"):
        export(dataset, output, lambda path, label: SkippedImage("unusable image"))
    assert not output.exists()
    assert len(pd.read_csv(failure_report_path(output))) == 4


@pytest.mark.parametrize("require_class_images", [False, True])
def test_empty_class_policy(dataset, tmp_path, require_class_images):
    (dataset / "b" / "z.PNG").unlink()
    (dataset / "b" / "a.tiff").unlink()
    output = tmp_path / "example.csv"
    if require_class_images:
        with pytest.raises(ValueError, match="No supported images found in"):
            export(dataset, output, lambda path, label: [1], require_class_images=True)
        assert not output.exists()
    else:
        frame = export(dataset, output, lambda path, label: [1])
        assert frame.Class.tolist() == ["a", "a"]


def test_empty_selection_uses_family_error_message(tmp_path):
    from microalgae.imaging.features._export import export_feature_table

    output = tmp_path / "example.csv"
    with pytest.raises(ValueError, match="No images selected"):
        export_feature_table(tmp_path, [], output, ["value"], lambda path, label: [1],
                             empty_message="No images selected")
    assert not output.exists()
    assert pd.read_csv(failure_report_path(output)).empty
