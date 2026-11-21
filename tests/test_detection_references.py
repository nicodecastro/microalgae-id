"""Detection annotation geometry, persistence, evaluation coverage and desktop workflow."""

from copy import deepcopy
import csv
import json

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QMessageBox

from scripts._detection_references import DetectionSession, digest
from scripts.annotate_detection import DetectionWindow


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / "raw"
    for label in ("a", "b"):
        (root / label).mkdir(parents=True)
        for i in range(3):
            image = np.full((80, 120, 3), 210, np.uint8)
            cv2.ellipse(image, (60, 40), (22, 9), 0, 0, 360, (40, 60, 50), -1)
            assert cv2.imwrite(str(root / label / f"{i}.png"), image)
    return root


@pytest.fixture
def session(sources, tmp_path):
    return DetectionSession.create(sources, tmp_path / "references", per_class=1)


def object_box(**overrides):
    return {"object_id": "first", "box": [10, 20, 90, 60], "species": "unknown",
            "kind": "cell", "uncertain": False, "truncated": False, "notes": "", **overrides}


def test_selection_deterministic_and_sources_untouched(sources, tmp_path):
    hashes = {p: digest(p) for p in sources.rglob("*.png")}
    a = DetectionSession.create(sources, tmp_path / "one", per_class=2, seed=42)
    b = DetectionSession.create(sources, tmp_path / "two", per_class=2, seed=42)
    assert a.data == b.data
    assert len(a.data["images"]) == 4
    assert not a.data["selection"]["explicit_list"]
    assert all(r["status"] == "pending" and not r["objects"] for r in a.data["images"])
    assert hashes == {p: digest(p) for p in hashes}
    with pytest.raises(FileExistsError):
        DetectionSession.create(sources, tmp_path / "one")


def test_roundtrip_backup_export_and_empty_coverage(session, tmp_path):
    before = session.path.read_bytes()
    row = deepcopy(session.data["images"][0])
    row.update(status="reviewed", partition="development", objects=[object_box(), object_box(object_id="mixed", kind="unresolved_group", uncertain=True)])
    session.save(0, row)
    assert (session.directory / "annotations.previous.json").read_bytes() == before
    row = deepcopy(session.data["images"][1])
    row.update(status="reviewed", partition="evaluation", empty_confirmed=True)
    session.save(1, row)
    resumed = DetectionSession(session.directory)
    assert resumed.data == session.data
    output = resumed.export(tmp_path / "snapshot")
    images = list(csv.DictReader((output / "images.csv").open()))
    boxes = list(csv.DictReader((output / "boxes.csv").open()))
    assert len(images) == 2 and images[1]["object_count"] == "0"
    assert [r["ignore"] for r in boxes] == ["False", "True"]
    assert all(r["parent_image_id"] == r["image_id"] for r in boxes)
    assert boxes[0]["right"] == "90"
    assert json.loads((output / "annotations.json").read_text()) == session.data
    with pytest.raises(FileExistsError):
        resumed.export(output)


@pytest.mark.parametrize("patch", [
    {"box": [10, 20, 10, 30]}, {"box": [-1, 0, 20, 30]},
    {"box": [0, 0, 121, 80]}, {"box": [0, 0, 10.5, 20]},
    {"box": [False, 0, 10, 20]}, {"species": "folder_label"},
    {"kind": "unresolved_group", "uncertain": False}, {"uncertain": "yes"},
])
def test_invalid_objects_never_replace_saved_annotations(session, patch):
    before = session.path.read_bytes()
    row = deepcopy(session.data["images"][0])
    row["objects"] = [object_box(**patch)]
    with pytest.raises(ValueError):
        session.save(0, row)
    assert session.path.read_bytes() == before
    assert not session.data["images"][0]["objects"]


def test_reviewed_empty_needs_confirmation_and_source_identity_is_fixed(session):
    row = deepcopy(session.data["images"][0])
    row["status"] = "reviewed"
    with pytest.raises(ValueError, match="Confirm"):
        session.save(0, row)
    row["empty_confirmed"] = True
    row["width"] = 999
    with pytest.raises(ValueError, match="source identity"):
        session.save(0, row)


def test_changed_source_blocks_resume_save_and_export(session, tmp_path):
    row = session.data["images"][0]
    path = session.root / row["image_id"]
    path.write_bytes(path.read_bytes() + b"changed")
    for operation in (lambda: DetectionSession(session.directory), lambda: session.save(0, row), lambda: session.export(tmp_path / "export")):
        with pytest.raises(ValueError, match="Source image changed"):
            operation()
    assert not (tmp_path / "export").exists()


def test_stale_window_cannot_overwrite_newer_annotations(session, tmp_path):
    stale = DetectionSession(session.directory)
    row = deepcopy(session.data["images"][0])
    row["notes"] = "new work"
    session.save(0, row)
    with pytest.raises(ValueError, match="another window"):
        stale.save(0, stale.data["images"][0])
    with pytest.raises(ValueError, match="another window"):
        stale.export(tmp_path / "stale")
    assert json.loads(session.path.read_text())["images"][0]["notes"] == "new work"


def test_selection_rejects_escape_and_outputs_inside_sources(sources, tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        DetectionSession.create(sources, tmp_path / "bad", image_ids=["../outside.png"])
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="outside"):
        DetectionSession.create(sources, sources / "output")
    with pytest.raises(ValueError, match="canonical"):
        DetectionSession.create(sources, tmp_path / "alias", image_ids=["a/../a/0.png"])


def test_pending_and_uncertain_images_are_not_silent_negatives(session, tmp_path):
    row = deepcopy(session.data["images"][0])
    row.update(objects=[object_box()], status="uncertain")
    session.save(0, row)
    output = session.export(tmp_path / "partial")
    boxes = list(csv.DictReader((output / "boxes.csv").open()))
    images = list(csv.DictReader((output / "images.csv").open()))
    assert boxes[0]["ignore"] == "True"
    assert images[1]["status"] == "pending" and images[1]["empty_confirmed"] == "False"


def test_failed_atomic_replace_keeps_last_saved_file(session, monkeypatch):
    from pathlib import Path
    before = session.path.read_bytes()
    row = deepcopy(session.data["images"][0])
    row["notes"] = "unsaved"
    def fail(*args):
        raise OSError("disk error")
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="disk error"):
        session.save(0, row)
    assert session.path.read_bytes() == before
    assert session.data["images"][0]["notes"] == ""


def test_desktop_draw_edit_save_resume_and_keep_zoom(qtbot, session):
    window = DetectionWindow(session)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: window.viewer.viewport().width() > 100)
    view = window.viewer
    window.activateWindow()
    view.setFocus()
    qtbot.waitUntil(window.isActiveWindow)
    qtbot.keyClick(view, Qt.Key.Key_D)
    assert view.drawing
    start = view.mapFromScene(QPointF(20, 25))
    end = view.mapFromScene(QPointF(95, 60))
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
    qtbot.mouseMove(view.viewport(), start)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    assert len(window.row["objects"]) == 1
    assert window.row["objects"][0]["species"] == "unknown"
    assert all(abs(a - b) <= 1 for a, b in zip(window.row["objects"][0]["box"], [20, 25, 95, 60]))
    view.zoom(1.25)
    transform = view.transform()
    window.species.setCurrentText("n_palea")
    window.kind.setCurrentText("colony_group")
    assert window.apply_object()
    assert view.transform() == transform
    window.review_next()
    assert session.data["images"][0]["status"] == "reviewed"
    assert session.data["images"][0]["objects"][0]["species"] == "n_palea"
    assert window.index == 1
    window.empty.setChecked(True)
    window.review_next()
    assert DetectionSession(session.directory).data["images"][1]["empty_confirmed"]


def test_crop_review_shortcuts_in_detection_window(qtbot, session, monkeypatch):
    window = DetectionWindow(session)
    monkeypatch.setattr(window, "can_leave", lambda: True)
    qtbot.addWidget(window)
    window.show()
    window.activateWindow()
    view = window.viewer
    view.setFocus()
    qtbot.waitUntil(window.isActiveWindow)
    window.add_box([5, 5, 25, 25])
    window.add_box([30, 30, 60, 60])

    window.object_notes.setText("first edit")
    window.object_changed()
    qtbot.keyClick(view, Qt.Key.Key_CapsLock)
    assert window.objects.currentRow() == 0
    assert not window.preview.pixmap().isNull()
    qtbot.keyClick(view, Qt.Key.Key_CapsLock)
    assert window.objects.currentRow() == 1
    assert window.object_notes.text() == "first edit"
    obj = window.row["objects"][1]
    qtbot.keyClick(view, Qt.Key.Key_N)
    qtbot.keyClick(view, Qt.Key.Key_U)
    assert window.row["objects"][1]["species"] == "other_alga"
    assert window.row["objects"][1]["uncertain"]
    qtbot.keyClick(view, Qt.Key.Key_A)
    assert not window.row["objects"][1]["uncertain"]
    qtbot.keyClick(view, Qt.Key.Key_B)
    assert view.drawing
    qtbot.keyClick(view, Qt.Key.Key_Escape)
    assert not view.drawing
    assert window.row["objects"][1]["box"] == [30, 30, 60, 60]
    qtbot.keyClick(view, Qt.Key.Key_B)
    view.box_drawn.emit([32, 32, 64, 64])
    assert not view.drawing
    assert len(window.row["objects"]) == 2
    assert window.row["objects"][1] == {
        **obj, "box": [32, 32, 64, 64], "species": "other_alga",
    }

    window.object_notes.setFocus()
    qtbot.keyClicks(window.object_notes, "brand")
    assert "brand" in window.object_notes.text()
    assert len(window.row["objects"]) == 2
    view.setFocus()
    qtbot.keyClick(view, Qt.Key.Key_CapsLock)
    qtbot.keyClick(view, Qt.Key.Key_R)
    assert len(window.row["objects"]) == 1
    qtbot.keyClick(view, Qt.Key.Key_M)
    assert session.data["images"][0]["status"] == "reviewed"
    assert window.index == 0
    qtbot.keyClick(view, Qt.Key.Key_PageDown)
    assert window.index == 1
    qtbot.keyClick(view, Qt.Key.Key_PageUp)
    assert window.index == 0


def test_navigation_cancel_and_discard_preserve_saved_data(qtbot, session, monkeypatch):
    window = DetectionWindow(session)
    qtbot.addWidget(window)
    window.add_box([2, 3, 20, 30])
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Cancel)
    window.images.setCurrentRow(1)
    assert window.index == 0 and window.images.currentRow() == 0
    assert not session.data["images"][0]["objects"]
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Discard)
    window.navigate(1)
    window.navigate(0)
    assert not window.row["objects"] and not window.dirty


def test_invalid_editor_blocks_save_and_selection_change(qtbot, session):
    window = DetectionWindow(session)
    qtbot.addWidget(window)
    window.add_box([2, 3, 20, 30])
    window.add_box([30, 3, 50, 30])
    window.coords[2].setValue(1)
    assert not window.save()
    window.objects.setCurrentRow(0)
    assert window.objects.currentRow() == 1
    assert "Invalid box" in window.statusBar().currentMessage()
    window.dirty = False
