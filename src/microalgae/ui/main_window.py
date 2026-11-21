"""Main offline image-review window."""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QByteArray, QEvent, QSize, Qt, QStandardPaths, QThreadPool, QTimer, QUrl, Slot
from PySide6.QtGui import (
    QAction,
    QDesktopServices,
    QIcon,
    QImage,
    QImageReader,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QBoxLayout,
    QFileDialog,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from microalgae.domain.results import ImageAnalysis
from microalgae.ui.image_viewer import ImageViewer
from microalgae.ui.models import ModelSelection
from microalgae.ui.predictions import PredictionTask, image_analysis
from microalgae.ui.results_panel import ResultsPanel


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
ZOOM_OUT_ICON = "M21 21l-4.4-4.4 M19 11a8 8 0 1 1-16 0a8 8 0 0 1 16 0 M8 11h6"
ENTER_FULLSCREEN_ICON = "M14 4h6v6 M20 4l-6 6 M4 14v6h6 M4 20l6-6"
EXIT_FULLSCREEN_ICON = "M20 4l-6 6 M14 4v6h6 M4 20l6-6 M4 14h6v6"


def _image_tool_icon(strokes: str, color: str = "#41516b") -> QIcon:
    svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24">
            <path d="{strokes}" fill="none" stroke="{color}" stroke-width="2"
                  stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    """
    pixmap = QPixmap()
    pixmap.loadFromData(svg.encode(), "SVG")
    return QIcon(pixmap)


def _image_navigation_icon(strokes: str) -> QIcon:
    icon = _image_tool_icon(strokes, "#17233a")
    disabled = _image_tool_icon(strokes, "#334155")
    icon.addPixmap(disabled.pixmap(48, 48), QIcon.Mode.Disabled)
    return icon


def _image_tool_button(label: str, strokes: str) -> QToolButton:
    button = QToolButton()
    button.setObjectName("imageHeaderTool")
    button.setIcon(_image_tool_icon(strokes))
    button.setIconSize(QSize(18, 18))
    button.setFixedSize(32, 32)
    button.setToolTip(label)
    button.setAccessibleName(label)
    return button


class _ImageHeader(QWidget):
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        direction = (
            QBoxLayout.Direction.TopToBottom
            if event.size().width() < 520
            else QBoxLayout.Direction.LeftToRight
        )
        self.layout().setDirection(direction)


class _ImageStatusRow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(2)

        heading = QHBoxLayout()
        self.message_label = QLabel()
        self.message_label.setObjectName("imageStatusMessage")
        self.message_label.setTextFormat(Qt.TextFormat.PlainText)
        self.message_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.message_label.setMinimumHeight(20)
        self._message = ""
        self.message_label.installEventFilter(self)
        heading.addWidget(self.message_label, 1)

        self.dismiss_button = QPushButton("Dismiss")
        self.dismiss_button.setObjectName("dismissImageStatus")
        self.dismiss_button.setFixedHeight(20)
        self.dismiss_button.clicked.connect(self._show_ready)
        heading.addWidget(self.dismiss_button)
        layout.addLayout(heading)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("imageStatusProgress")
        self.progress_bar.setAccessibleName("Prediction run progress")
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(2)
        policy = self.progress_bar.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.progress_bar.setSizePolicy(policy)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._show_ready)
        self._show_ready()

    def set_message(self, message: str, timeout_ms: int = 0, *, dismissible: bool = False) -> None:
        self.timer.stop()
        self._message = message
        self.dismiss_button.setVisible(dismissible)
        self.message_label.setAccessibleName(message)
        self._render_message()
        if timeout_ms:
            self.timer.start(timeout_ms)

    def _show_ready(self) -> None:
        self.set_message("Ready. Images stay on this computer.")

    def _render_message(self) -> None:
        label = self.message_label
        elided = label.fontMetrics().elidedText(
            self._message, Qt.TextElideMode.ElideRight, label.contentsRect().width()
        )
        label.setText(elided)
        label.setToolTip(self._message if elided != self._message else "")

    def eventFilter(self, watched, event) -> bool:
        if watched is self.message_label and event.type() == QEvent.Type.Resize:
            self._render_message()
        return super().eventFilter(watched, event)


@dataclass(slots=True)
class ReviewImage:
    image_id: str
    title: str
    path: Path | None = None
    analysis: ImageAnalysis | None = None
    prediction_directory: Path | None = None


@dataclass(frozen=True, slots=True)
class ImportReport:
    added: int
    duplicates: int
    errors: tuple[str, ...]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MicroalgaeID")
        self.resize(1440, 850)
        self.setMinimumSize(900, 500)
        self.setAcceptDrops(True)

        self._images: list[ReviewImage] = []
        self._path_ids: dict[str, str] = {}
        self.models = ModelSelection()
        self.prediction_task = None
        self._prediction_entries: list[ReviewImage] = []
        self.prediction_directory = None
        self.prediction_rows = {}
        self._build_toolbar()
        self._build_content()
        crop_menu = self.menuBar().addMenu("Crop review")
        crop_menu.addAction("Open review batch", self.open_crop_batch)
        crop_menu.addAction("Detect imported image", lambda: self.detect_imported())
        crop_menu.addAction("Detect field-test image", lambda: self.detect_imported("evaluation"))
        menu = self.menuBar().addMenu("Predictions")
        self.open_prediction_action = menu.addAction("Open prediction run…", self.open_prediction_run)
        self.show_run_action = menu.addAction("Show run folder", self.show_run_folder)
        self.models.add_menu(self, show_status=self._show_status_message)
        offline_label = QLabel("OFFLINE")
        offline_label.setObjectName("statusBadge")
        self.statusBar().addPermanentWidget(offline_label)
        self._update_actions()

    @staticmethod
    def prediction_folder():
        return Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)) / "predictions"

    def _show_status_message(self, message: str, timeout_ms: int = 8000) -> None:
        if self.prediction_task is None and self.status_row.dismiss_button.isHidden():
            self.status_row.set_message(message, timeout_ms)

    def run_predictions(self) -> None:
        self._run_predictions(self._images)

    def run_current_prediction(self) -> None:
        row = self.image_list.currentRow()
        if 0 <= row < len(self._images):
            self._run_predictions([self._images[row]])

    def _run_predictions(self, entries: list[ReviewImage]) -> None:
        entries = [entry for entry in entries if entry.path is not None]
        if not entries or self.prediction_task is not None:
            return
        settings = self.models.settings
        saved_model_path = settings.value("classifier_path", "", str)
        model_path = saved_model_path or self.models.choose_classifier_path(self)
        if not model_path:
            return

        directory = self.prediction_folder() / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        images = [(entry.image_id, entry.path) for entry in entries]
        self._prediction_entries = entries
        self._prediction_saved_model_path = saved_model_path
        self.prediction_task = PredictionTask(
            directory, images, detector_path=settings.value("detector", "", str),
            classifier=self.models.classifier, model_path=model_path,
            approved_fingerprint=settings.value("classifier_fingerprint", "", str),
        )
        self.prediction_task.trust_requested.connect(self.confirm_prediction_classifier)
        self.prediction_task.prepared.connect(self.prediction_prepared)
        self.prediction_task.progress.connect(self.prediction_progress)
        self.prediction_task.finished.connect(self.prediction_finished)
        self._update_actions()
        self.status_row.set_message("Preparing predictions…")
        self.status_row.progress_bar.setRange(0, 0)
        self.status_row.progress_bar.show()
        QThreadPool.globalInstance().start(self.prediction_task.run)

    @Slot(object)
    def confirm_prediction_classifier(self, path):
        task = self.prediction_task
        if not task.stop.is_set():
            task.answer_trust(self.models.confirm_classifier(self, path))

    @Slot(object)
    def prediction_prepared(self, classifier):
        task = self.prediction_task
        if (task.classifier is None and self.models.classifier is None
                and self.models.settings.value("classifier_path", "", str) == self._prediction_saved_model_path):
            self.models.remember_classifier(task.model_path, classifier)
        self.prediction_directory = task.directory
        for entry in self._prediction_entries:
            self.prediction_rows.pop(entry.image_id, None)
            entry.prediction_directory = self.prediction_directory
            entry.analysis = ImageAnalysis(entry.image_id, (), "model", "Waiting for prediction")
        self._show_row(self.image_list.currentRow())

    @Slot(int, object)
    def prediction_progress(self, index, row):
        progress_bar = self.status_row.progress_bar
        if row is None:
            total = len(self._prediction_entries)
            progress_bar.setRange(0, total)
            progress_bar.setValue(index - 1)
            if not self.prediction_task.stop.is_set():
                self.status_row.set_message(f"Prediction run · Processing image {index} of {total}")
        else:
            self.refresh_prediction(row)
            if row["status"] != "stopped":
                progress_bar.setValue(index)

    @Slot(object)
    def prediction_finished(self, error):
        stopped = self.prediction_task.stop.is_set()
        started = self.prediction_directory == self.prediction_task.directory
        self.prediction_task = None
        if started:
            for entry in self._prediction_entries:
                if entry.image_id not in self.prediction_rows:
                    entry.analysis = ImageAnalysis(entry.image_id, (), "model", "Not processed")
            self._show_row(self.image_list.currentRow())
        else:
            self._update_actions()
        self._prediction_entries.clear()
        self.status_row.progress_bar.hide()
        if error is not None:
            title = "Prediction run interrupted" if started else "Could not run predictions"
            self.status_row.set_message(title, dismissible=True)
            QMessageBox.warning(self, title, error)
        else:
            self._show_status_message("Prediction run stopped" if stopped else "Predictions saved")

    def stop_predictions(self):
        if self.prediction_task is not None:
            self.prediction_task.cancel()
            self.stop_predictions_action.setEnabled(False)
            self.status_row.set_message("Stopping after the current operation…")

    def open_prediction_run(self):
        directory = QFileDialog.getExistingDirectory(self, "Open prediction run", str(self.prediction_folder()))
        if not directory:
            return
        try:
            from microalgae.imaging.crop_review import CropBatch

            batch = CropBatch(directory)
            if batch.data["collection"] != "prediction":
                raise ValueError("Choose a saved prediction run.")
        except Exception as error:
            QMessageBox.warning(self, "Could not open prediction run", str(error))
            return
        self.clear_session()
        self.prediction_directory = batch.directory
        for row in batch.data["images"]:
            identifier = row["image_id"]
            source = batch.directory / row["source_path"] if row.get("source_path") else None
            self.prediction_rows[identifier] = row
            self._images.append(ReviewImage(
                identifier, row["title"], source,
                analysis=image_analysis(row), prediction_directory=batch.directory,
            ))
            if source is not None:
                self._path_ids[os.path.normcase(str(source.resolve()))] = identifier
            item = QListWidgetItem(row["title"])
            item.setData(Qt.ItemDataRole.UserRole, identifier)
            self.image_list.addItem(item)
        self.image_list.setCurrentRow(0)
        self._update_actions()

    def show_run_folder(self):
        if self.prediction_directory is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.prediction_directory)))

    @Slot(object)
    def refresh_prediction(self, row):
        self.prediction_rows[row["image_id"]] = row
        for index, entry in enumerate(self._images):
            if entry.image_id == row["image_id"]:
                entry.analysis = image_analysis(row)
                if index == self.image_list.currentRow():
                    self._show_row(index)
                break

    def edit_prediction_crops(self):
        from microalgae.imaging.crop_review import CropBatch
        from microalgae.ui.crop_review import CropReviewWindow

        row = self.image_list.currentRow()
        if row < 0 or not self.edit_crops_action.isEnabled():
            return
        try:
            entry = self._images[row]
            batch = CropBatch(entry.prediction_directory)
            index = next(i for i, item in enumerate(batch.data["images"]) if item["image_id"] == entry.image_id)
            self.crop_window = CropReviewWindow(batch, self, models=self.models, initial_image=index)
            self.crop_window.setWindowModality(Qt.WindowModality.WindowModal)
            self.crop_window.prediction_changed.connect(self.refresh_prediction)
            self.crop_window.show()
        except Exception as error:
            QMessageBox.warning(self, "Could not edit crops", str(error))

    def closeEvent(self, event):
        if self.prediction_task is not None:
            self.stop_predictions()
            event.ignore()
        else:
            super().closeEvent(event)

    def open_crop_batch(self):
        directory = QFileDialog.getExistingDirectory(self, "Open crop review batch")
        if directory:
            try:
                from microalgae.imaging.crop_review import CropBatch
                from microalgae.ui.crop_review import CropReviewWindow

                self.crop_window = CropReviewWindow(CropBatch(directory), self, models=self.models)
                self.crop_window.show()
            except Exception as error:
                QMessageBox.warning(self, "Crop review", str(error))

    def detect_imported(self, collection="training"):
        from microalgae.imaging.crop_review import CropBatch
        from microalgae.ui.crop_review import CropReviewWindow
        index = self.image_list.currentRow()
        if index < 0 or self._images[index].path is None:
            return
        path = self._images[index].path
        destination, _ = QFileDialog.getSaveFileName(self, "New review directory")
        if destination:
            try:
                batch = CropBatch.prepare(
                    path.parent, destination, settings=self.models.detector_settings(),
                    files=[path], collection=collection,
                )
                self.crop_window = CropReviewWindow(batch, self, models=self.models)
                self.crop_window.show()
            except Exception as error:
                QMessageBox.warning(self, "Detection", str(error))

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Image review", self)
        toolbar.setMovable(False)
        toolbar.setObjectName("mainToolbar")
        self.addToolBar(toolbar)
        self.main_toolbar = toolbar

        self.open_action = QAction("Add images", self)
        self.open_action.setShortcut(QKeySequence("Ctrl+O"))
        self.open_action.triggered.connect(self.open_files_dialog)
        toolbar.addAction(self.open_action)
        self.run_predictions_action = toolbar.addAction("Run predictions", self.run_predictions)
        self.stop_predictions_action = toolbar.addAction("Stop predictions", self.stop_predictions)
        self.edit_crops_action = toolbar.addAction("Edit crops", self.edit_prediction_crops)

        toolbar.addSeparator()

        self.previous_action = QAction(_image_navigation_icon("M15.5 5l-7 7 7 7"), "Previous", self)
        self.previous_action.setShortcuts([QKeySequence(Qt.Key.Key_Left), QKeySequence("Alt+Left")])
        self.previous_action.triggered.connect(self.previous_image)
        toolbar.addAction(self.previous_action)
        toolbar.widgetForAction(self.previous_action).setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        self.next_action = QAction(_image_navigation_icon("M8.5 5l7 7-7 7"), "Next", self)
        self.next_action.setShortcuts([QKeySequence(Qt.Key.Key_Right), QKeySequence("Alt+Right")])
        self.next_action.triggered.connect(self.next_image)
        toolbar.addAction(self.next_action)
        toolbar.widgetForAction(self.next_action).setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        toolbar.addSeparator()

        self.zoom_out_action = QAction("−", self)
        self.zoom_out_action.setToolTip("Zoom out")
        self.zoom_out_action.triggered.connect(lambda: self.viewer.zoom(1 / 1.25))
        toolbar.addAction(self.zoom_out_action)

        self.zoom_in_action = QAction("+", self)
        self.zoom_in_action.setToolTip("Zoom in")
        self.zoom_in_action.triggered.connect(lambda: self.viewer.zoom(1.25))
        toolbar.addAction(self.zoom_in_action)

        self.fit_action = QAction("Fit image", self)
        self.fit_action.setShortcut(QKeySequence("Ctrl+0"))
        self.fit_action.triggered.connect(lambda: self.viewer.fit_image())
        toolbar.addAction(self.fit_action)
        toolbar.addSeparator()

        self.annotations_action = QAction("Show annotations", self)
        self.annotations_action.setCheckable(True)
        self.annotations_action.setChecked(True)
        toolbar.addAction(self.annotations_action)

        self.clear_action = QAction("Clear session", self)
        self.clear_action.triggered.connect(self.clear_session)
        toolbar.addAction(self.clear_action)

    def _build_content(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setObjectName("mainSplitter")
        self.setCentralWidget(splitter)

        left = QWidget()
        left.setObjectName("sidePanel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 18)
        left_layout.setSpacing(0)

        brand_header = QHBoxLayout()
        brand_header.setContentsMargins(18, 20, 18, 20)
        brand_header.setSpacing(8)
        microscope = QSvgWidget()
        microscope.load(QByteArray(b"""
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">
                <rect width="40" height="40" rx="10" fill="#dbeafe"/>
                <g transform="translate(8 8)" fill="none" stroke="#2164e8"
                   stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 22h18 M6 18h7 M14 7a7.5 7.5 0 0 1 0 15 M10 13v2 M8 6V2h4v4"/>
                    <rect x="7" y="6" width="6" height="7" rx="1"/>
                </g>
            </svg>
        """))
        microscope.setFixedSize(36, 36)
        brand_header.addWidget(microscope, 0, Qt.AlignmentFlag.AlignVCenter)

        brand_text = QVBoxLayout()
        brand_text.setSpacing(2)
        brand = QLabel("MicroalgaeID")
        brand.setObjectName("brandTitle")
        brand_text.addWidget(brand)
        description = QLabel("Automated Species Classification")
        description.setObjectName("brandSubtitle")
        brand_text.addWidget(description)
        brand_header.addLayout(brand_text, 1)
        left_layout.addLayout(brand_header)

        separator = QWidget()
        separator.setObjectName("panelDivider")
        separator.setFixedHeight(1)
        left_layout.addWidget(separator)

        images_layout = QVBoxLayout()
        images_layout.setContentsMargins(18, 12, 18, 0)
        images_layout.setSpacing(12)
        left_layout.addLayout(images_layout, 1)

        images_header = QHBoxLayout()
        images_title = QLabel("Images")
        images_title.setObjectName("sectionTitle")
        images_header.addWidget(images_title, 1)

        add_button = QPushButton("Add images")
        add_button.setObjectName("primaryButton")
        add_button.clicked.connect(self.open_files_dialog)
        images_header.addWidget(add_button)
        images_layout.addLayout(images_header)

        self.image_list = QListWidget()
        self.image_list.setObjectName("imageList")
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.image_list.currentRowChanged.connect(self._show_row)
        images_layout.addWidget(self.image_list, 1)

        session_note = QLabel("This list clears when the app closes.")
        session_note.setObjectName("mutedText")
        session_note.setWordWrap(True)
        images_layout.addWidget(session_note)
        splitter.addWidget(left)

        center = QWidget()
        center.setObjectName("centerPanel")
        center.setMinimumWidth(320)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 10, 0, 0)
        center_layout.setSpacing(12)
        status_section = QVBoxLayout()
        status_section.setSpacing(4)
        self.status_row = _ImageStatusRow()
        status_section.addWidget(self.status_row)
        status_divider = QWidget()
        status_divider.setObjectName("panelDivider")
        status_divider.setFixedHeight(1)
        status_section.addWidget(status_divider)
        center_layout.addLayout(status_section)

        image_header = _ImageHeader()
        header = QHBoxLayout(image_header)
        header.setContentsMargins(20, 0, 20, 0)
        header.setSpacing(8)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.image_title = QLabel("Microscopy image")
        self.image_title.setObjectName("sectionTitle")
        self.image_title.setTextFormat(Qt.TextFormat.PlainText)
        self.image_title.setWordWrap(True)
        titles.addWidget(self.image_title)
        self.image_detail = QLabel("Choose a local image")
        self.image_detail.setObjectName("mutedText")
        self.image_detail.setWordWrap(True)
        titles.addWidget(self.image_detail)
        header.addLayout(titles, 1)
        header.addLayout(self._make_image_header_tools())
        center_layout.addWidget(image_header)

        self.center_stack = QStackedWidget()
        self.empty_page = self._make_empty_page()
        self.error_page = self._make_error_page()
        self.viewer = ImageViewer()
        self.viewer.detection_selected.connect(self._on_box_selected)
        self.viewer.files_dropped.connect(self.add_files)
        self.annotations_action.toggled.connect(self.viewer.set_annotations_visible)
        self.center_stack.addWidget(self.empty_page)
        self.center_stack.addWidget(self.viewer)
        self.center_stack.addWidget(self.error_page)
        self.center_stack.setCurrentWidget(self.empty_page)

        image_area = QGridLayout()
        image_area.addWidget(self.center_stack, 0, 0)
        navigation = QHBoxLayout()
        navigation.setContentsMargins(12, 0, 12, 0)
        self.previous_image_button = QToolButton()
        self.previous_image_button.setDefaultAction(self.previous_action)
        self.previous_image_button.setAccessibleName("Previous image")
        self.previous_image_button.setObjectName("imageNavigation")
        self.previous_image_button.setIconSize(QSize(22, 22))
        self.previous_image_button.setFixedSize(48, 48)
        navigation.addWidget(self.previous_image_button, 0, Qt.AlignmentFlag.AlignVCenter)
        navigation.addStretch(1)

        self.next_image_button = QToolButton()
        self.next_image_button.setDefaultAction(self.next_action)
        self.next_image_button.setAccessibleName("Next image")
        self.next_image_button.setObjectName("imageNavigation")
        self.next_image_button.setIconSize(QSize(22, 22))
        self.next_image_button.setFixedSize(48, 48)
        navigation.addWidget(self.next_image_button, 0, Qt.AlignmentFlag.AlignVCenter)
        image_area.addLayout(navigation, 0, 0)
        center_layout.addLayout(image_area, 1)
        splitter.addWidget(center)

        splitter.addWidget(self._make_results_pane())
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(1, False)
        splitter.setSizes([260, 995, 265])

    def _make_results_pane(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("sidePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        controls = QWidget()
        controls.setObjectName("predictionControls")
        buttons = QVBoxLayout(controls)
        buttons.setContentsMargins(18, 20, 18, 16)
        buttons.setSpacing(8)
        self.predict_current_button = QPushButton("Predict current image")
        self.predict_current_button.setObjectName("predictCurrentButton")
        self.predict_current_button.clicked.connect(self.run_current_prediction)
        buttons.addWidget(self.predict_current_button)
        self.predict_all_button = QPushButton("Predict all images")
        self.predict_all_button.clicked.connect(self.run_predictions)
        buttons.addWidget(self.predict_all_button)
        layout.addWidget(controls)

        separator = QWidget()
        separator.setObjectName("panelDivider")
        separator.setFixedHeight(1)
        layout.addWidget(separator)

        self.results_panel = ResultsPanel()
        self.results_panel.detection_selected.connect(self.viewer.select_detection)
        layout.addWidget(self.results_panel, 1)
        return panel

    def _make_image_header_tools(self) -> QHBoxLayout:
        tools = QHBoxLayout()
        tools.setSpacing(4)
        tools.addStretch(1)
        self.annotations_button = QPushButton("Show annotations")
        self.annotations_button.setObjectName("imageAnnotationsButton")
        self.annotations_button.clicked.connect(self.annotations_action.trigger)
        self.annotations_action.changed.connect(self._sync_annotations_button)
        tools.addWidget(self.annotations_button)

        self.zoom_out_button = _image_tool_button("Zoom out", ZOOM_OUT_ICON)
        self.zoom_out_button.clicked.connect(self.zoom_out_action.trigger)
        tools.addWidget(self.zoom_out_button)
        self.zoom_in_button = _image_tool_button("Zoom in", f"{ZOOM_OUT_ICON} M11 8v6")
        self.zoom_in_button.clicked.connect(self.zoom_in_action.trigger)
        tools.addWidget(self.zoom_in_button)

        self.fullscreen_button = _image_tool_button("Full screen", ENTER_FULLSCREEN_ICON)
        self.fullscreen_button.setCheckable(True)
        self.fullscreen_button.toggled.connect(self._set_image_fullscreen)
        tools.addWidget(self.fullscreen_button)

        self.exit_fullscreen_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.exit_fullscreen_shortcut.setEnabled(False)
        self.exit_fullscreen_shortcut.activated.connect(lambda: self.fullscreen_button.setChecked(False))
        return tools

    def _sync_annotations_button(self) -> None:
        available = self.annotations_action.isEnabled()
        visible = available and self.annotations_action.isChecked()
        self.annotations_button.setEnabled(available)
        self.annotations_button.setText("Hide annotations" if visible else "Show annotations")

    def _set_image_fullscreen(self, enabled: bool) -> None:
        splitter = self.centralWidget()
        if enabled:
            self._fullscreen_splitter_sizes = splitter.sizes()
            self._fullscreen_was_maximized = self.isMaximized()
            chrome = (splitter.widget(0), splitter.widget(2), self.main_toolbar, self.menuBar(), self.statusBar())
            self._fullscreen_hidden_widgets = [widget for widget in chrome if not widget.isHidden()]
            for widget in self._fullscreen_hidden_widgets:
                widget.hide()
            self.showFullScreen()
        else:
            if self._fullscreen_was_maximized:
                self.showMaximized()
            else:
                self.showNormal()
            for widget in self._fullscreen_hidden_widgets:
                widget.show()
            splitter.setSizes(self._fullscreen_splitter_sizes)

        label = "Exit full screen" if enabled else "Full screen"
        self.fullscreen_button.setAccessibleName(label)
        self.fullscreen_button.setToolTip("Exit full screen (Esc)" if enabled else label)
        self.fullscreen_button.setIcon(_image_tool_icon(EXIT_FULLSCREEN_ICON if enabled else ENTER_FULLSCREEN_ICON))
        self.exit_fullscreen_shortcut.setEnabled(enabled)
        self.fullscreen_button.setEnabled(enabled or self.center_stack.currentWidget() is self.viewer)

    def _make_empty_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("emptyPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.addStretch(1)
        heading = QLabel("Review microscopy images offline")
        heading.setObjectName("emptyTitle")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(heading)
        body = QLabel("Drag and drop local images here to get started.")
        body.setObjectName("mutedText")
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(body)
        add_button = QPushButton("Add images")
        add_button.setObjectName("primaryButton")
        add_button.clicked.connect(self.open_files_dialog)
        layout.addWidget(add_button, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)
        return page

    def _make_error_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("emptyPage")
        layout = QVBoxLayout(page)
        layout.addStretch(1)
        self.error_message = QLabel()
        self.error_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_message.setWordWrap(True)
        layout.addWidget(self.error_message)
        layout.addStretch(1)
        return page

    def open_files_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add microscopy images", str(Path.home()), IMAGE_FILTER
        )
        if paths:
            self.add_files(paths)

    def add_files(self, paths, *, show_errors: bool = True) -> ImportReport:
        added = 0
        duplicates = 0
        errors: list[str] = []
        first_new_row: int | None = None
        for raw_path in paths:
            try:
                path = Path(raw_path).expanduser().resolve()
            except (OSError, RuntimeError) as error:
                errors.append(f"{raw_path}: {error}")
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                errors.append(f"{path.name}: unsupported format")
                continue
            if not path.is_file():
                errors.append(f"{path.name}: file not found")
                continue
            key = os.path.normcase(str(path))
            if key in self._path_ids:
                duplicates += 1
                continue

            reader = QImageReader(str(path))
            reader.setAutoTransform(True)
            if not reader.canRead():
                errors.append(f"{path.name}: {reader.errorString() or 'cannot read image'}")
                continue
            if path.suffix.lower() in {".tif", ".tiff"} and reader.imageCount() > 1:
                errors.append(f"{path.name}: multi-page TIFF is not supported")
                continue
            if reader.read().isNull():
                errors.append(f"{path.name}: {reader.errorString() or 'cannot decode image'}")
                continue

            image_id = uuid4().hex
            entry = ReviewImage(image_id=image_id, title=path.name, path=path)
            if first_new_row is None:
                first_new_row = len(self._images)
            self._images.append(entry)
            self._path_ids[key] = image_id
            list_item = QListWidgetItem(path.name)
            list_item.setData(Qt.ItemDataRole.UserRole, image_id)
            list_item.setToolTip(str(path))
            self.image_list.addItem(list_item)
            added += 1

        if first_new_row is not None:
            self.image_list.setCurrentRow(first_new_row)
        self._show_status_message(
            f"Added {added} image(s); skipped {duplicates} duplicate(s) "
            f"and {len(errors)} unreadable/unsupported file(s).",
            8000,
        )
        if errors and show_errors:
            details = "\n".join(errors[:8])
            if len(errors) > 8:
                details += f"\n…and {len(errors) - 8} more"
            QMessageBox.warning(self, "Some images were skipped", details)
        self._update_actions()
        return ImportReport(added, duplicates, tuple(errors))

    def clear_session(self) -> None:
        self.prediction_directory = None
        self.prediction_rows.clear()
        self._images.clear()
        self._path_ids.clear()
        self.image_list.clear()
        self.viewer.clear_image()
        self.center_stack.setCurrentWidget(self.empty_page)
        self.results_panel.set_analysis(None, has_image=False)
        self.image_title.setText("Microscopy image")
        self.image_detail.setText("Choose a local image")
        self._show_status_message("Session cleared. Original files were not changed.")
        self._update_actions()

    def previous_image(self) -> None:
        row = self.image_list.currentRow()
        if row > 0:
            self.image_list.setCurrentRow(row - 1)

    def next_image(self) -> None:
        row = self.image_list.currentRow()
        if 0 <= row < len(self._images) - 1:
            self.image_list.setCurrentRow(row + 1)

    def _show_row(self, row: int) -> None:
        if not 0 <= row < len(self._images):
            self.center_stack.setCurrentWidget(self.empty_page)
            self.results_panel.set_analysis(None, has_image=False)
            self._update_actions()
            return

        entry = self._images[row]
        saved = self.prediction_rows.get(entry.image_id)
        source = entry.path
        auto_transform = True
        if entry.prediction_directory is not None and saved and saved.get("source_path"):
            source = entry.prediction_directory / saved["source_path"]
            auto_transform = False
        if source is not None:
            reader = QImageReader(str(source))
            reader.setAutoTransform(auto_transform)
            image = reader.read()
            error = reader.errorString()
        else:
            image = QImage()
            error = "Image was not processed."

        self.image_title.setText(entry.title)
        if image.isNull():
            self.viewer.clear_image()
            self.error_message.setText(f"Could not open this image.\n{error or 'File unavailable.'}")
            self.center_stack.setCurrentWidget(self.error_page)
            self.image_detail.setText(f"Image {row + 1} of {len(self._images)}")
            self.results_panel.set_analysis(entry.analysis, has_image=True)
        else:
            self.viewer.set_image(image, entry.analysis)
            self.viewer.set_annotations_visible(self.annotations_action.isChecked())
            self.center_stack.setCurrentWidget(self.viewer)
            kind = "Prediction run" if entry.analysis is not None else "Local image · no analysis"
            self.image_detail.setText(
                f"{kind} · {image.width()} × {image.height()} · "
                f"Image {row + 1} of {len(self._images)}"
            )
            self.results_panel.set_analysis(entry.analysis, has_image=True)
        self._update_actions()

    def _on_box_selected(self, identifier: str) -> None:
        self.results_panel.select_detection(identifier)

    def _update_actions(self) -> None:
        row = self.image_list.currentRow() if hasattr(self, "image_list") else -1
        selected = self._images[row] if 0 <= row < len(self._images) else None
        has_image = hasattr(self, "viewer") and self.center_stack.currentWidget() is self.viewer
        has_detections = (
            has_image
            and selected is not None
            and selected.analysis is not None
            and bool(selected.analysis.detections)
        )
        self.previous_action.setEnabled(row > 0)
        self.next_action.setEnabled(0 <= row < len(self._images) - 1)
        self.previous_image_button.setVisible(0 <= row < len(self._images))
        self.next_image_button.setVisible(0 <= row < len(self._images))
        self.zoom_in_action.setEnabled(has_image)
        self.zoom_out_action.setEnabled(has_image)
        self.zoom_in_button.setEnabled(has_image)
        self.zoom_out_button.setEnabled(has_image)
        self.fullscreen_button.setEnabled(has_image or self.fullscreen_button.isChecked())
        self.fit_action.setEnabled(has_image)
        self.annotations_action.setEnabled(has_detections)
        running = self.prediction_task is not None
        saved = self.prediction_rows.get(selected.image_id) if selected is not None else None
        can_predict_all = not running and any(entry.path is not None for entry in self._images)
        self.run_predictions_action.setEnabled(can_predict_all)
        self.predict_all_button.setEnabled(can_predict_all)
        self.predict_current_button.setEnabled(not running and selected is not None and selected.path is not None)
        self.stop_predictions_action.setVisible(running)
        self.stop_predictions_action.setEnabled(running and not self.prediction_task.stop.is_set())
        self.open_prediction_action.setEnabled(not running)
        self.show_run_action.setEnabled(self.prediction_directory is not None and self.prediction_directory.exists())
        self.edit_crops_action.setEnabled(
            not running and bool(saved and saved.get("size") and selected.prediction_directory)
        )
        self.clear_action.setEnabled(not running and bool(self._images))

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(
            url.isLocalFile() for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        self.add_files(paths)
        event.acceptProposedAction()
