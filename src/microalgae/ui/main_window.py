"""Main offline image-review window."""

from dataclasses import dataclass
import os
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QImage, QImageReader, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from microalgae.domain.results import ImageAnalysis
from microalgae.ui.demo import DEMO_IMAGE_ID, create_demo
from microalgae.ui.image_viewer import ImageViewer
from microalgae.ui.results_panel import ResultsPanel


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff)"


@dataclass(slots=True)
class ReviewImage:
    image_id: str
    title: str
    path: Path | None = None
    demo_image: QImage | None = None
    analysis: ImageAnalysis | None = None


@dataclass(frozen=True, slots=True)
class ImportReport:
    added: int
    duplicates: int
    errors: tuple[str, ...]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MicroalgaeID · Offline image review")
        self.resize(1440, 850)
        self.setMinimumSize(900, 500)
        self.setAcceptDrops(True)

        self._images: list[ReviewImage] = []
        self._path_ids: dict[str, str] = {}
        self._build_toolbar()
        self._build_content()
        self.statusBar().showMessage("Ready. Images stay on this computer.")
        offline_label = QLabel("OFFLINE · SESSION ONLY")
        offline_label.setObjectName("statusBadge")
        self.statusBar().addPermanentWidget(offline_label)
        self._update_actions()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Image review", self)
        toolbar.setMovable(False)
        toolbar.setObjectName("mainToolbar")
        self.addToolBar(toolbar)

        self.open_action = QAction("Add images", self)
        self.open_action.setShortcut(QKeySequence("Ctrl+O"))
        self.open_action.triggered.connect(self.open_files_dialog)
        toolbar.addAction(self.open_action)

        self.demo_action = QAction("Open demo", self)
        self.demo_action.triggered.connect(self.open_demo)
        toolbar.addAction(self.demo_action)
        toolbar.addSeparator()

        self.previous_action = QAction("Previous", self)
        self.previous_action.setShortcut(QKeySequence("Alt+Left"))
        self.previous_action.triggered.connect(self.previous_image)
        toolbar.addAction(self.previous_action)

        self.next_action = QAction("Next", self)
        self.next_action.setShortcut(QKeySequence("Alt+Right"))
        self.next_action.triggered.connect(self.next_image)
        toolbar.addAction(self.next_action)
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
        left.setMinimumWidth(180)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(18, 20, 18, 18)
        left_layout.setSpacing(12)
        brand = QLabel("MicroalgaeID")
        brand.setObjectName("brandTitle")
        left_layout.addWidget(brand)
        description = QLabel("Local microscopy image review")
        description.setObjectName("mutedText")
        description.setWordWrap(True)
        left_layout.addWidget(description)
        images_title = QLabel("Images")
        images_title.setObjectName("sectionTitle")
        left_layout.addWidget(images_title)

        self.image_list = QListWidget()
        self.image_list.setObjectName("imageList")
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.image_list.currentRowChanged.connect(self._show_row)
        left_layout.addWidget(self.image_list, 1)

        add_button = QPushButton("Add images")
        add_button.setObjectName("primaryButton")
        add_button.clicked.connect(self.open_files_dialog)
        left_layout.addWidget(add_button)
        demo_button = QPushButton("Open demo")
        demo_button.clicked.connect(self.open_demo)
        left_layout.addWidget(demo_button)
        session_note = QLabel("This list clears when the app closes.")
        session_note.setObjectName("mutedText")
        session_note.setWordWrap(True)
        left_layout.addWidget(session_note)
        splitter.addWidget(left)

        center = QWidget()
        center.setObjectName("centerPanel")
        center.setMinimumWidth(320)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(20, 18, 20, 18)
        center_layout.setSpacing(12)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        self.image_title = QLabel("Microscopy image")
        self.image_title.setObjectName("sectionTitle")
        titles.addWidget(self.image_title)
        self.image_detail = QLabel("Choose a local image or open the demo")
        self.image_detail.setObjectName("mutedText")
        titles.addWidget(self.image_detail)
        header.addLayout(titles, 1)
        self.zoom_label = QLabel("—")
        self.zoom_label.setObjectName("mutedText")
        header.addWidget(self.zoom_label)
        center_layout.addLayout(header)

        self.center_stack = QStackedWidget()
        self.empty_page = self._make_empty_page()
        self.error_page = self._make_error_page()
        self.viewer = ImageViewer()
        self.viewer.detection_selected.connect(self._on_box_selected)
        self.viewer.files_dropped.connect(self.add_files)
        self.viewer.zoom_changed.connect(lambda percent: self.zoom_label.setText(f"{percent}%"))
        self.annotations_action.toggled.connect(self.viewer.set_annotations_visible)
        self.center_stack.addWidget(self.empty_page)
        self.center_stack.addWidget(self.viewer)
        self.center_stack.addWidget(self.error_page)
        self.center_stack.setCurrentWidget(self.empty_page)
        center_layout.addWidget(self.center_stack, 1)
        splitter.addWidget(center)

        self.results_panel = ResultsPanel()
        self.results_panel.detection_selected.connect(self.viewer.select_detection)
        splitter.addWidget(self.results_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(1, False)
        splitter.setSizes([205, 995, 265])

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
        body = QLabel("Add local images to browse them, or open a clearly marked demo of results.")
        body.setObjectName("mutedText")
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(body)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        add_button = QPushButton("Add images")
        add_button.setObjectName("primaryButton")
        add_button.clicked.connect(self.open_files_dialog)
        buttons.addWidget(add_button)
        demo_button = QPushButton("Open demo")
        demo_button.clicked.connect(self.open_demo)
        buttons.addWidget(demo_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
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
        self.statusBar().showMessage(
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

    def open_demo(self) -> None:
        for row, entry in enumerate(self._images):
            if entry.image_id == DEMO_IMAGE_ID:
                self.image_list.setCurrentRow(row)
                return
        image, analysis = create_demo()
        self._images.append(
            ReviewImage(DEMO_IMAGE_ID, "Demo image · sample results", demo_image=image, analysis=analysis)
        )
        item = QListWidgetItem("DEMO IMAGE")
        item.setData(Qt.ItemDataRole.UserRole, DEMO_IMAGE_ID)
        item.setToolTip("Synthetic image with sample detections, not a model prediction")
        self.image_list.addItem(item)
        self.image_list.setCurrentRow(len(self._images) - 1)
        self.statusBar().showMessage("Demo opened. These are sample results only.", 8000)

    def clear_session(self) -> None:
        self._images.clear()
        self._path_ids.clear()
        self.image_list.clear()
        self.viewer.clear_image()
        self.center_stack.setCurrentWidget(self.empty_page)
        self.results_panel.set_analysis(None, has_image=False)
        self.image_title.setText("Microscopy image")
        self.image_detail.setText("Choose a local image or open the demo")
        self.zoom_label.setText("—")
        self.statusBar().showMessage("Session cleared. Original files were not changed.", 8000)
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
        if entry.demo_image is not None:
            image = entry.demo_image
            error = ""
        else:
            reader = QImageReader(str(entry.path))
            reader.setAutoTransform(True)
            image = reader.read()
            error = reader.errorString()

        self.image_title.setText(entry.title)
        if image.isNull():
            self.viewer.clear_image()
            self.error_message.setText(f"Could not open this image.\n{error or 'File unavailable.'}")
            self.center_stack.setCurrentWidget(self.error_page)
            self.image_detail.setText(f"Image {row + 1} of {len(self._images)}")
            self.results_panel.set_analysis(None, has_image=True)
            self.zoom_label.setText("—")
        else:
            self.viewer.set_image(image, entry.analysis)
            self.viewer.set_annotations_visible(self.annotations_action.isChecked())
            self.center_stack.setCurrentWidget(self.viewer)
            kind = "DEMO · sample data" if entry.analysis is not None else "Local image · no analysis"
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
        has_image = hasattr(self, "viewer") and self.center_stack.currentWidget() is self.viewer
        has_detections = (
            has_image
            and 0 <= row < len(self._images)
            and self._images[row].analysis is not None
            and bool(self._images[row].analysis.detections)
        )
        self.previous_action.setEnabled(row > 0)
        self.next_action.setEnabled(0 <= row < len(self._images) - 1)
        self.zoom_in_action.setEnabled(has_image)
        self.zoom_out_action.setEnabled(has_image)
        self.fit_action.setEnabled(has_image)
        self.annotations_action.setEnabled(has_detections)
        self.clear_action.setEnabled(bool(self._images))

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
