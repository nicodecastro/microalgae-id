"""Focused behavior checks for the offline desktop review prototype."""

import builtins
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from PIL import Image
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt, QMimeData, QSettings, QThread, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage
from PySide6.QtWidgets import QApplication, QFileDialog, QGraphicsPixmapItem, QMessageBox

from microalgae.domain.results import Detection, ImageAnalysis
from microalgae.ui.main_window import MainWindow
from microalgae.ui.results_panel import ResultsPanel


def _image_file(path: Path) -> Path:
    image = QImage(180, 120, QImage.Format.Format_RGB32)
    image.fill("#b8c6c4")
    assert image.save(str(path))
    return path


@pytest.fixture
def window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    with qtbot.waitExposed(window):
        window.show()
    return window


def _assert_background_click_keeps_image(window, qtbot, scene_point):
    viewer = window.viewer
    for _ in range(2):
        qtbot.mouseClick(
            viewer.viewport(), Qt.MouseButton.LeftButton,
            pos=viewer.mapFromScene(scene_point),
        )
        assert any(isinstance(item, QGraphicsPixmapItem) for item in viewer.scene().items())
        assert window.center_stack.currentWidget() is viewer


def _publish_prediction(window):
    image_id = window.image_list.currentItem().data(Qt.ItemDataRole.UserRole)
    window.refresh_prediction({
        "image_id": image_id,
        "status": "proposals",
        "error": None,
        "crops": [
            {
                "crop_id": "first",
                "active": True,
                "box": (30, 40, 65, 75),
                "classification": {"species": "Species A"},
            },
            {
                "crop_id": "second",
                "active": True,
                "box": (110, 58, 145, 94),
                "classification": {"species": "Species B"},
            },
        ],
    })


def test_image_review_and_session_reset(window, qtbot, tmp_path):
    assert window.image_list.count() == 0
    assert window.center_stack.currentWidget() is window.empty_page
    assert window.status_row.message_label.text() == "Ready. Images stay on this computer."

    window.add_files([_image_file(tmp_path / "review.png")])
    assert window.status_row.message_label.text().startswith("Added 1 image(s)")
    assert not window.statusBar().currentMessage()
    _publish_prediction(window)
    assert window.image_list.count() == 1
    assert window.results_panel.result_list.count() == 2
    assert "Prediction run" in window.image_detail.text()
    assert not window.results_panel.prediction_time.isVisible()
    assert len(window.viewer._overlay_items) == 2

    analysis = window._images[0].analysis
    detection = analysis.detections[0]
    viewer = window.viewer
    box = viewer._overlay_items[detection.identifier]
    rect = box.mapRectToScene(box.rect())
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (
        detection.x, detection.y, detection.width, detection.height
    )

    initial_zoom = viewer.transform().m11()
    window.zoom_in_button.click()
    assert viewer.transform().m11() > initial_zoom
    assert box.mapRectToScene(box.rect()) == rect
    window.zoom_out_button.click()
    assert viewer.transform().m11() == pytest.approx(initial_zoom)

    results = window.results_panel.result_list
    results.setCurrentRow(1)
    assert viewer._selected_id == analysis.detections[1].identifier
    point = viewer.mapFromScene(rect.center())
    qtbot.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton, pos=point)
    assert viewer._selected_id == detection.identifier
    assert results.currentItem().data(Qt.ItemDataRole.UserRole) == detection.identifier
    _assert_background_click_keeps_image(window, qtbot, QPointF(90, 20))

    assert window.annotations_button.text() == "Hide annotations"
    window.annotations_button.click()
    assert not box.isVisible()
    assert window.annotations_button.text() == "Show annotations"
    window.annotations_action.trigger()
    assert box.isVisible()
    assert window.annotations_button.text() == "Hide annotations"
    window.fit_action.trigger()
    assert box.mapRectToScene(box.rect()) == rect

    window.fullscreen_button.click()
    assert window.isFullScreen() and not window.results_panel.isVisible()
    assert window.status_row.isVisible()
    qtbot.waitUntil(window.isActiveWindow)
    qtbot.keyClick(viewer, Qt.Key.Key_Escape)
    assert not window.isFullScreen() and window.results_panel.isVisible()

    window.clear_session()
    assert window.image_list.count() == 0
    assert window.center_stack.currentWidget() is window.empty_page
    assert results.count() == 0
    assert window.status_row.message_label.text() == "Session cleared. Original files were not changed."


def test_cancel_open_review_batch_without_loading_crop_review(window, monkeypatch):
    events = []
    original_import = builtins.__import__

    def track_import(name, *args, **kwargs):
        if name in {"microalgae.imaging.crop_review", "microalgae.ui.crop_review"}:
            events.append(name)
        return original_import(name, *args, **kwargs)

    def cancel_folder_picker(*args):
        events.append("folder picker")
        return ""

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", cancel_folder_picker)
    crop_menu = window.menuBar().actions()[0].menu()
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", track_import)
        crop_menu.actions()[0].trigger()

    assert events == ["folder picker"]


def test_import_and_navigation_keep_results_per_image(window, qtbot, tmp_path):
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

    report = window.add_files(
        [first, second, first, bad, unsupported, multi_page, tiff], show_errors=False
    )
    assert report.added == 3
    assert report.duplicates == 1
    assert len(report.errors) == 3
    assert any("multi-page TIFF" in error for error in report.errors)
    assert window.image_list.count() == 3
    assert window.image_list.currentRow() == 0
    assert not window.previous_image_button.isEnabled()
    assert window.center_stack.currentWidget() is window.viewer
    _publish_prediction(window)
    assert window.results_panel.result_list.count() == 2

    qtbot.waitUntil(window.isActiveWindow)
    window.viewer.setFocus()
    qtbot.keyClick(window.viewer, Qt.Key.Key_Right)
    assert window.image_list.currentRow() == 1
    assert not window.viewer._overlay_items
    assert window.results_panel.result_list.count() == 0
    assert "not available yet" in window.results_panel.message.text()
    assert not window.results_panel.summary.isVisible()
    _assert_background_click_keeps_image(window, qtbot, QPointF(50, 50))

    qtbot.keyClick(window.viewer, Qt.Key.Key_Left)
    assert window.image_list.currentRow() == 0
    assert window.results_panel.result_list.count() == 2
    assert not window.previous_image_button.isEnabled()
    window.next_image_button.click()
    assert window.image_list.currentRow() == 1
    qtbot.keyClick(window.viewer, Qt.Key.Key_Right)
    assert window.image_list.currentRow() == 2
    assert not window.next_image_button.isEnabled()
    window.previous_image_button.click()
    assert window.image_list.currentRow() == 1
    assert first.read_bytes() == first_bytes


def test_prediction_buttons_keep_each_images_saved_run(window, qtbot, tmp_path, monkeypatch):
    from microalgae.imaging import prediction
    from microalgae.imaging.crop_review import CropBatch

    detector = SimpleNamespace(detect=lambda _: [{"box": [30, 40, 65, 75], "crop_box": [30, 40, 65, 75]}])
    classifier = SimpleNamespace(name="Test model", fingerprint="test", predict=lambda _: "Species A")
    monkeypatch.setattr(prediction, "make_detector", lambda _: detector)
    readings = iter((10.0, 10.125, 20.0, 22.5, 30.0, 31.25))
    monkeypatch.setattr(prediction, "perf_counter", readings.__next__)
    window.models.settings = QSettings(str(tmp_path / "models.ini"), QSettings.Format.IniFormat)
    window.models.remember_classifier(str(tmp_path / "classifier"), classifier)
    monkeypatch.setattr(window, "prediction_folder", lambda: tmp_path / "predictions")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _, title, message: warnings.append((title, message)))

    assert not window.predict_current_button.isEnabled()
    assert not window.predict_all_button.isEnabled()
    first = _image_file(tmp_path / "first.png")
    second = _image_file(tmp_path / "second.png")
    window.add_files([first, second])
    window.predict_all_button.click()
    assert not window.predict_all_button.isEnabled()
    assert not window.predict_current_button.isEnabled()
    qtbot.waitUntil(lambda: window.prediction_task is None)
    all_run = CropBatch(window.prediction_directory)
    assert [row["title"] for row in all_run.data["images"]] == [first.name, second.name]
    assert [row["prediction_seconds"] for row in all_run.data["images"]] == [0.125, 2.5]
    assert window.results_panel.prediction_time.text() == "Predicted in 125 ms"

    window.image_list.setCurrentRow(1)
    assert window.results_panel.prediction_time.text() == "Predicted in 2.50 s"
    window.predict_current_button.click()
    qtbot.waitUntil(lambda: window.prediction_task is None)
    current_run = CropBatch(window.prediction_directory)
    assert current_run.directory != all_run.directory
    assert [row["title"] for row in current_run.data["images"]] == [second.name]
    assert window.results_panel.prediction_time.text() == "Predicted in 1.25 s"

    first.unlink()
    window.image_list.setCurrentRow(0)
    assert window.center_stack.currentWidget() is window.viewer
    assert "Species A" in window.results_panel.result_list.item(0).text()
    assert window.results_panel.prediction_time.text() == "Predicted in 125 ms"
    window.edit_crops_action.trigger()
    assert not warnings
    qtbot.addWidget(window.crop_window)
    assert window.crop_window.batch.directory == all_run.directory
    window.crop_window.close()

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_: str(all_run.directory))
    window.open_prediction_run()
    window.image_list.setCurrentRow(1)
    assert window.results_panel.prediction_time.text() == "Predicted in 2.50 s"


@pytest.mark.parametrize("decision", ["approve", "decline", "stop", "load_error"])
def test_prediction_preparation_keeps_ui_responsive(window, qtbot, tmp_path, monkeypatch, decision):
    from microalgae.imaging import classifier, prediction

    app_thread = window.thread()
    loading = Event()
    release_load = Event()
    model_path = str(tmp_path / "classifier")
    model = SimpleNamespace(name="Test model", fingerprint="test")
    window.models.settings = QSettings(str(tmp_path / "models.ini"), QSettings.Format.IniFormat)
    if decision == "approve":
        def choose_folder(*args):
            assert QThread.currentThread() == app_thread
            return model_path

        monkeypatch.setattr(QFileDialog, "getExistingDirectory", choose_folder)
    else:
        window.models.settings.setValue("classifier_path", model_path)
    monkeypatch.setattr(window, "prediction_folder", lambda: tmp_path / "predictions")
    monkeypatch.setattr(prediction, "make_detector", lambda _: SimpleNamespace(detect=lambda _: []))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _, title, message: warnings.append((title, message)))
    prompts = []

    def load(path, approved, confirm):
        assert QThread.currentThread() != app_thread
        assert path == model_path
        assert approved == ""
        loading.set()
        assert release_load.wait(5)
        if decision == "load_error":
            raise ValueError("Invalid classifier")
        if not confirm(Path(path) / "pipeline.joblib"):
            raise classifier.ClassifierSelectionCancelled()
        return model

    def question(*args):
        assert QThread.currentThread() == app_thread
        prompts.append(args[1])
        return QMessageBox.StandardButton.Yes if decision == "approve" else QMessageBox.StandardButton.No

    monkeypatch.setattr(classifier, "load_classifier", load)
    monkeypatch.setattr(QMessageBox, "question", question)
    window.add_files([_image_file(tmp_path / "image.png")])
    _publish_prediction(window)
    previous_rows = window.prediction_rows.copy()
    window.predict_all_button.click()
    status = window.status_row
    try:
        assert status.message_label.text() == "Preparing predictions…"
        assert status.progress_bar.isVisible() and status.progress_bar.maximum() == 0
        qtbot.waitUntil(loading.is_set)
        assert window.prediction_rows == previous_rows
        if decision == "stop":
            window.stop_predictions_action.trigger()
            assert status.message_label.text() == "Stopping after the current operation…"
    finally:
        release_load.set()
    qtbot.waitUntil(lambda: window.prediction_task is None)

    assert warnings == ([("Could not run predictions", "Invalid classifier")] if decision == "load_error" else [])
    assert prompts == (["Trust this classifier?"] if decision in ("approve", "decline") else [])
    if decision == "approve":
        assert window.models.classifier is model
        assert window.models.settings.value("classifier_path") == model_path
        assert window.models.settings.value("classifier_fingerprint") == model.fingerprint
        assert next(iter(window.prediction_rows.values()))["status"] == "zero proposals"
    else:
        assert window.models.classifier is None
        assert window.prediction_rows == previous_rows
        assert window.prediction_directory is None
    assert window.predict_all_button.isEnabled()
    assert not status.progress_bar.isVisible()
    if decision == "load_error":
        assert status.message_label.text() == "Could not run predictions"
        assert status.dismiss_button.isVisible() and not status.timer.isActive()
        status.dismiss_button.click()
    else:
        expected = "Predictions saved" if decision == "approve" else "Prediction run stopped"
        assert status.message_label.text() == expected
        assert status.timer.isActive() and status.timer.interval() == 8000
        status.timer.setInterval(1)
    qtbot.waitUntil(lambda: status.message_label.text() == "Ready. Images stay on this computer.")


def test_prediction_progress_survives_browsing_imports_and_stop(window, qtbot, tmp_path, monkeypatch):
    from microalgae.imaging import prediction

    begin_second_image = Event()

    def run(_directory, images, _settings, _classifier, _model_path, stop, progress):
        first, second = (image_id for image_id, _ in images)
        progress(1, None)
        progress(1, {"image_id": first, "status": "failed", "error": "Could not read", "crops": []})
        assert begin_second_image.wait(5)
        progress(2, None)
        assert stop.wait(5)
        progress(2, {"image_id": second, "status": "stopped", "error": None, "crops": []})

    monkeypatch.setattr(prediction, "run_predictions", run)
    monkeypatch.setattr(window, "prediction_folder", lambda: tmp_path / "predictions")
    window.models.settings = QSettings(str(tmp_path / "models.ini"), QSettings.Format.IniFormat)
    classifier = SimpleNamespace(name="Test model", fingerprint="test")
    window.models.remember_classifier(str(tmp_path / "classifier"), classifier)
    window.add_files([_image_file(tmp_path / name) for name in ("first.png", "second.png")])
    window.image_list.setCurrentRow(1)
    window.predict_all_button.click()
    status = window.status_row
    try:
        qtbot.waitUntil(lambda: status.progress_bar.value() == 1)
        assert status.progress_bar.maximum() == 2
        assert status.message_label.text() == "Prediction run · Processing image 1 of 2"
        begin_second_image.set()
        processing = "Prediction run · Processing image 2 of 2"
        qtbot.waitUntil(lambda: status.message_label.text() == processing)
        window.previous_image()
        window.add_files([_image_file(tmp_path / "third.png")])
        assert window.image_list.count() == 3
        assert status.message_label.text() == processing
        window.stop_predictions_action.trigger()
        assert status.message_label.text() == "Stopping after the current operation…"
    finally:
        begin_second_image.set()
        if window.prediction_task is not None:
            window.stop_predictions()

    qtbot.waitUntil(lambda: window.prediction_task is None)
    assert status.message_label.text() == "Prediction run stopped"
    assert status.progress_bar.value() == 1 and not status.progress_bar.isVisible()


def test_drop_local_file_and_missing_file_state(window, tmp_path):
    path = _image_file(tmp_path / "drop.png")
    second_path = _image_file(tmp_path / "drop-on-viewer.png")
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
    detection = Detection("a", 0, 0, 20, 20, "Species A", probability=0.94)
    analysis = ImageAnalysis(
        image_id="model-image",
        source="model",
        detections=(detection,),
    )
    panel.set_analysis(analysis, has_image=True)
    assert "94%" not in panel.result_list.item(0).text()
    assert "—" in panel.summary.text()

    validated = replace(detection, probability_validated=True)
    panel.set_analysis(replace(analysis, detections=(validated,)), has_image=True)
    assert "94%" in panel.result_list.item(0).text()
    assert "94.0%" in panel.summary.text()
