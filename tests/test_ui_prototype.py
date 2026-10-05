"""Focused behavior checks for the offline desktop review prototype."""

from pathlib import Path

from PIL import Image
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt, QMimeData, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem

from microalgae.domain.results import Detection, ImageAnalysis
from microalgae.ui.main_window import MainWindow
from microalgae.ui.results_panel import ResultsPanel


def _image_file(path: Path) -> Path:
    image = QImage(180, 120, QImage.Format.Format_RGB32)
    image.fill("#b8c6c4")
    assert image.save(str(path))
    return path


def test_empty_demo_and_session_reset(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.image_list.count() == 0
    assert window.center_stack.currentWidget() is window.empty_page

    window.open_demo()
    window.open_demo()
    assert window.image_list.count() == 1
    assert window.results_panel.result_list.count() == 2
    assert "DEMO" in window.image_detail.text()
    assert window.results_panel.demo_badge.isVisible()
    assert len(window.viewer._overlay_items) == 2

    window.clear_session()
    assert window.image_list.count() == 0
    assert window.center_stack.currentWidget() is window.empty_page
    assert window.results_panel.result_list.count() == 0
    window.close()

    fresh_window = MainWindow()
    qtbot.addWidget(fresh_window)
    assert fresh_window.image_list.count() == 0


def test_import_navigation_skips_duplicates_and_bad_files(qtbot, tmp_path):
    first = _image_file(tmp_path / "first.png")
    second = _image_file(tmp_path / "second.jpg")
    tiff = _image_file(tmp_path / "third.tiff")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("hello")
    multi_page = tmp_path / "pages.tiff"
    Image.new("RGB", (12, 12), "red").save(
        multi_page,
        save_all=True,
        append_images=[Image.new("RGB", (12, 12), "blue")],
    )
    first_bytes = first.read_bytes()

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    report = window.add_files(
        [first, second, first, bad, unsupported, multi_page, tiff], show_errors=False
    )
    assert report.added == 3
    assert report.duplicates == 1
    assert len(report.errors) == 3
    assert any("multi-page TIFF" in error for error in report.errors)
    assert window.image_list.count() == 3
    assert window.image_list.currentRow() == 0
    assert window.center_stack.currentWidget() is window.viewer
    assert not window.viewer._overlay_items
    assert "not available yet" in window.results_panel.message.text()
    assert not window.results_panel.summary.isVisible()

    window.next_image()
    assert window.image_list.currentRow() == 1
    window.next_image()
    assert window.image_list.currentRow() == 2
    window.previous_image()
    assert window.image_list.currentRow() == 1
    assert first.read_bytes() == first_bytes


def test_demo_boxes_cards_zoom_and_toggle_stay_synchronized(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_demo()
    qtbot.wait(20)
    analysis = window._images[0].analysis
    detection = analysis.detections[0]
    box = window.viewer._overlay_items[detection.identifier]
    rect = box.mapRectToScene(box.rect())
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (
        detection.x, detection.y, detection.width, detection.height
    )

    window.viewer.zoom(1.25)
    assert box.mapRectToScene(box.rect()) == rect


    card = window.results_panel.result_list.item(1)
    window.results_panel.result_list.setCurrentItem(card)
    assert window.viewer._selected_id == analysis.detections[1].identifier

    point = window.viewer.mapFromScene(
        QPointF(detection.x + detection.width / 2, detection.y + detection.height / 2)
    )
    qtbot.mouseClick(window.viewer.viewport(), Qt.MouseButton.LeftButton, pos=point)
    assert window.viewer._selected_id == detection.identifier
    assert window.results_panel.result_list.currentItem().data(Qt.ItemDataRole.UserRole) == (
        detection.identifier
    )

    window.annotations_action.setChecked(False)
    assert not box.isVisible()
    window.annotations_action.setChecked(True)
    assert box.isVisible()
    window.viewer.fit_image()
    assert box.mapRectToScene(box.rect()) == rect


@pytest.mark.parametrize("demo", [False, True])
def test_clicking_image_does_not_remove_it(qtbot, tmp_path, demo):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    if demo:
        window.open_demo()
        scene_point = QPointF(500, 100)
    else:
        window.add_files([_image_file(tmp_path / "imported.png")])
        scene_point = QPointF(50, 50)

    viewer = window.viewer
    qtbot.wait(20)
    for _ in range(2):
        qtbot.mouseClick(
            viewer.viewport(), Qt.MouseButton.LeftButton,
            pos=viewer.mapFromScene(scene_point),
        )
        assert any(isinstance(item, QGraphicsPixmapItem) for item in viewer.scene().items())
        assert window.center_stack.currentWidget() is viewer


def test_imported_image_never_inherits_demo_results(qtbot, tmp_path):
    path = _image_file(tmp_path / "user.png")
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_demo()
    window.add_files([path])
    assert window.image_list.currentRow() == 1
    assert window.results_panel.result_list.count() == 0
    assert not window.results_panel.demo_badge.isVisible()
    assert not window.viewer._overlay_items
    window.previous_image()
    assert window.results_panel.result_list.count() == 2


def test_drop_local_file_and_missing_file_state(qtbot, tmp_path):
    path = _image_file(tmp_path / "drop.png")
    second_path = _image_file(tmp_path / "drop-on-viewer.png")
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    drop = QDropEvent(
        QPointF(30, 30),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.dropEvent(drop)
    assert window.image_list.count() == 1

    viewer_mime = QMimeData()
    viewer_mime.setUrls([QUrl.fromLocalFile(str(second_path))])
    enter = QDragEnterEvent(
        QPoint(30, 30), Qt.DropAction.CopyAction, viewer_mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(window.viewer.viewport(), enter)
    viewer_drop = QDropEvent(
        QPointF(30, 30), Qt.DropAction.CopyAction, viewer_mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(window.viewer.viewport(), viewer_drop)
    assert enter.isAccepted() and viewer_drop.isAccepted()
    assert window.image_list.count() == 2

    path.unlink()
    window.image_list.setCurrentRow(-1)
    window.image_list.setCurrentRow(0)
    assert window.center_stack.currentWidget() is window.error_page
    assert "Could not open" in window.error_message.text()


def test_model_probabilities_require_explicit_validation(qtbot):
    panel = ResultsPanel()
    qtbot.addWidget(panel)
    analysis = ImageAnalysis(
        image_id="model-image",
        source="model",
        detections=(Detection("a", 0, 0, 20, 20, "Species A", probability=0.94),),
    )
    panel.set_analysis(analysis, has_image=True)
    assert "94%" not in panel.result_list.item(0).text()
    assert "—" in panel.summary.text()

    validated = ImageAnalysis(
        image_id="model-image",
        source="model",
        detections=(
            Detection(
                "a", 0, 0, 20, 20, "Species A",
                probability=0.94, probability_validated=True,
            ),
        ),
    )
    panel.set_analysis(validated, has_image=True)
    assert "94%" in panel.result_list.item(0).text()
    assert "94.0%" in panel.summary.text()
