"""Expandable annotation sessions preserve human masks and provenance."""

import json
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import yaml

from scripts._geometry_references import ReferenceSession, digest, mask_path
from scripts.annotate_geometry import AnnotationWindow, main
from scripts.score_geometry_masks import score_geometry_masks
from test_geometry_references import comparison, POLYGONS


def select_first(comparison, count):
    path = comparison / "reference_selection.csv"
    frame = pd.read_csv(path)
    frame["selected"] = False
    frame.loc[frame.index[:count], "selected"] = True
    frame.to_csv(path, index=False)


@pytest.mark.parametrize("count", [1, 5, 12])
def test_arbitrary_count_dynamic_title_and_legacy_resume(comparison, tmp_path, count):
    import matplotlib.pyplot as plt

    select_first(comparison, count)
    session = ReferenceSession(comparison, tmp_path / "references")
    assert len(session.data["images"]) == count
    window = AnnotationWindow(session)
    try:
        assert f"1/{count}" in window.axes.get_title()
    finally:
        plt.close(window.figure)
    # Missing selection_mode is the legacy CSV-selection format.
    del session.data["selection_mode"]
    session.path.write_text(json.dumps(session.data))
    assert ReferenceSession(comparison, session.output, resume=True).data == session.data


def test_all_selection_import_resume_and_scoring(comparison, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    select_first(comparison, 3)
    source = ReferenceSession(comparison, tmp_path / "source")
    source.save(0, POLYGONS, "approved")
    source.save(1, [], "uncertain")
    before = {p: digest(p) for p in source.output.rglob("*") if p.is_file()}
    choices = (comparison / "reference_selection.csv").read_bytes()
    destination = tmp_path / "expanded"
    monkeypatch.setattr("scripts.annotate_geometry.AnnotationWindow", lambda *args: pytest.fail("GUI opened"))
    assert main(["--comparison-dir", str(comparison), "--output-dir", str(destination),
                 "--select-all", "--import-references", str(source.output), "--prepare-only"]) == 0
    session = ReferenceSession(comparison, destination, resume=True)
    assert session.data["selection_mode"] == "all"
    assert len(session.data["images"]) == 12
    assert [r["status"] for r in session.data["images"]].count("pending") == 10
    assert session.data["images"][:3] == source.data["images"]
    assert session.data["imported_from"] == {"directory": str(source.output), "annotations_sha256": digest(source.path)}
    for row in source.data["images"][:2]:
        assert mask_path(destination, row["image_id"]).read_bytes() == mask_path(source.output, row["image_id"]).read_bytes()
    assert len(list(destination.rglob("*_mask.png"))) == 2
    assert (comparison / "reference_selection.csv").read_bytes() == choices
    assert {p: digest(p) for p in before} == before
    # All-selection resume is independent of the comparison's CSV choices.
    select_first(comparison, 0)
    resumed = ReferenceSession(comparison, destination, resume=True)
    assert resumed.data == session.data
    window = AnnotationWindow(resumed)
    try:
        assert window.index == 2 and "3/12" in window.axes.get_title()
    finally:
        plt.close(window.figure)
    assert main(["--comparison-dir", str(comparison), "--output-dir", str(destination), "--resume", "--prepare-only"]) == 0
    scores = score_geometry_masks(comparison, destination, tmp_path / "scores")
    assert len(scores) == 6
    exclusions = pd.read_csv(tmp_path / "scores" / "excluded_references.csv")
    assert len(exclusions) == 11 and exclusions.status.value_counts().to_dict() == {"pending": 10, "uncertain": 1}


@pytest.mark.parametrize("problem", ["baseline", "label", "hash", "shape", "mask", "uncertain_mask", "outside_selection"])
def test_invalid_import_creates_no_output(comparison, tmp_path, problem):
    select_first(comparison, 1)
    source = ReferenceSession(comparison, tmp_path / "source")
    source.save(0, POLYGONS, "approved" if problem != "uncertain_mask" else "uncertain")
    target = tmp_path / "target_comparison"
    shutil.copytree(comparison, target)
    if problem == "baseline":
        path = target / "resolved_settings.yaml"
        settings = yaml.safe_load(path.read_text())
        settings["profiles"]["canny_baseline"]["filter"]["enabled"] = True
        path.write_text(yaml.safe_dump(settings))
    elif problem == "label":
        path = target / "selection.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "Class"] = "wrong"
        frame.to_csv(path, index=False)
    elif problem in ("hash", "shape"):
        path = target / "source_images.json"
        inventory = json.loads(path.read_text())
        inventory[0]["sha256" if problem == "hash" else "canvas_shape"] = "wrong" if problem == "hash" else [1, 1]
        path.write_text(json.dumps(inventory))
    elif problem in ("mask", "uncertain_mask"):
        assert cv2.imwrite(str(mask_path(source.output, "a/0.png")), np.zeros((64, 64), dtype=np.uint8))
    else:
        choices = pd.read_csv(target / "reference_selection.csv")
        choices["selected"] = choices.image_id.eq("a/1.png")
        choices.to_csv(target / "reference_selection.csv", index=False)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        ReferenceSession(target, output, select_all=problem != "outside_selection", import_references=source.output)
    assert not output.exists()


def test_creation_options_rejected_on_resume_and_existing_output_preserved(comparison, tmp_path):
    source = ReferenceSession(comparison, tmp_path / "source", select_all=True)
    original = source.path.read_bytes()
    with pytest.raises(FileExistsError):
        ReferenceSession(comparison, source.output, select_all=True)
    for options in ({"select_all": True}, {"import_references": source.output}):
        with pytest.raises(ValueError, match="only when creating"):
            ReferenceSession(comparison, source.output, resume=True, **options)
    with pytest.raises(ValueError, match="outside the source"):
        ReferenceSession(comparison, source.output / "nested", select_all=True, import_references=source.output)
    assert source.path.read_bytes() == original
    assert not (source.output / "nested").exists()
