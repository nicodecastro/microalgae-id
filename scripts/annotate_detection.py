"""Draw and review original-resolution cell/colony boxes in an offline desktop window."""

import argparse
from copy import deepcopy
from math import ceil, floor
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import cv2
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QFontDatabase, QImage, QKeySequence, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QGraphicsSimpleTextItem, QLabel, QLineEdit, QListWidget,
    QMainWindow, QMessageBox, QPushButton, QSpinBox, QSplitter, QToolBar,
    QVBoxLayout, QWidget,
)

from microalgae.ui.app import STYLE
from microalgae.ui.image_viewer import ImageViewer
from scripts._detection_references import (
    DetectionSession, KINDS, PARTITIONS, SPECIES, STATUSES, read_image,
)


def configure_app(app):
    # Sandboxed/offscreen Windows sessions may not enumerate system fonts.
    if not QFontDatabase.families():
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLE + "\nQPushButton:disabled { color: #7a879a; background: #f4f6f9; border-color: #dce2eb; }")


class BoxViewer(ImageViewer):
    box_drawn = Signal(list)

    def __init__(self):
        super().__init__()
        self.drawing = False
        self.start = None
        self.preview = None

    def refresh_boxes(self, objects, selected=None):
        for box in self._overlay_items.values():
            self.scene().removeItem(box)
        self._overlay_items.clear()
        for number, obj in enumerate(objects, 1):
            left, top, right, bottom = obj["box"]
            box = self.scene().addRect(QRectF(left, top, right - left, bottom - top))
            box.setZValue(1)
            marker = QGraphicsSimpleTextItem(str(number), box)
            marker.setBrush(QColor("#ffffff"))
            marker.setPos(left, top)
            marker.setFlag(marker.GraphicsItemFlag.ItemIgnoresTransformations)
            self._overlay_items[obj["object_id"]] = box
        self.select_detection(selected)

    def cancel_drawing(self):
        if self.preview is not None:
            self.scene().removeItem(self.preview)
        self.start = self.preview = None

    def mousePressEvent(self, event):
        if self.drawing and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(point):
                self.start = point
                pen = QPen(QColor("#f59e0b"), 2)
                pen.setCosmetic(True)
                self.preview = self.scene().addRect(QRectF(point, point), pen)
                self.preview.setZValue(2)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.start is not None:
            point = self.mapToScene(event.position().toPoint())
            self.preview.setRect(QRectF(self.start, point).normalized().intersected(self.sceneRect()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.start is not None and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            rect = QRectF(self.start, point).normalized().intersected(self.sceneRect())
            self.scene().removeItem(self.preview)
            self.start = self.preview = None
            if rect.width() >= 1 and rect.height() >= 1:
                self.box_drawn.emit([floor(rect.left()), floor(rect.top()), ceil(rect.right()), ceil(rect.bottom())])
            event.accept()
            return
        super().mouseReleaseEvent(event)


class DetectionWindow(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.index = next((i for i, row in enumerate(session.data["images"]) if row["status"] != "reviewed"), 0)
        self.dirty = False
        self.updating = False
        self.setWindowTitle("MicroalgaeID · Detection references")
        self.resize(1380, 860)
        self.setMinimumSize(1050, 700)
        toolbar = QToolBar("Annotation", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.draw_action = QAction("Draw boxes", self)
        self.draw_action.setCheckable(True)
        self.draw_action.setShortcut(QKeySequence("B"))
        self.draw_action.toggled.connect(self.set_drawing)
        toolbar.addAction(self.draw_action)
        self.action(toolbar, "Fit image", "Ctrl+0", lambda: self.viewer.fit_image())
        self.action(toolbar, "Save", "Ctrl+S", self.save)
        self.action(toolbar, "Previous", "Alt+Left", lambda: self.navigate(self.index - 1))
        self.action(toolbar, "Next", "Alt+Right", lambda: self.navigate(self.index + 1))
        self.action(toolbar, "Export snapshot", "Ctrl+E", self.export)

        splitter = QSplitter()
        self.setCentralWidget(splitter)
        sidebar = QWidget()
        sidebar.setObjectName("sidePanel")
        sidebar.setMinimumWidth(230)
        left = QVBoxLayout(sidebar)
        heading = QLabel("Detection reference set")
        heading.setObjectName("sectionTitle")
        left.addWidget(heading)
        self.progress = QLabel()
        self.progress.setWordWrap(True)
        left.addWidget(self.progress)
        self.images = QListWidget()
        self.images.setAccessibleName("Source images and saved review status")
        self.images.setObjectName("imageList")
        self.images.currentRowChanged.connect(self.navigate)
        left.addWidget(self.images)
        hint = QLabel("Annotate every identifiable alga.\nKeep each colony intact.\nExclude rulers, text and debris.\n\nFolder names are not object labels.")
        hint.setWordWrap(True)
        left.addWidget(hint)
        splitter.addWidget(sidebar)

        center = QWidget()
        layout = QVBoxLayout(center)
        self.title = QLabel()
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self.viewer = BoxViewer()
        self.viewer.setAccessibleName("Image annotation canvas")
        self.viewer.box_drawn.connect(self.add_box)
        self.viewer.detection_selected.connect(self.select_box)
        layout.addWidget(self.viewer, 1)
        layout.addWidget(QLabel("B: draw boxes · turn off to select/pan · wheel: zoom · Ctrl+0: fit"))
        splitter.addWidget(center)

        panel = QWidget()
        panel.setObjectName("sidePanel")
        panel.setMinimumWidth(290)
        right = QVBoxLayout(panel)
        right.addWidget(QLabel("Objects in this image"))
        self.objects = QListWidget()
        self.objects.setAccessibleName("Cell and colony boxes")
        self.objects.setObjectName("resultList")
        self.objects.setMaximumHeight(165)
        self.objects.currentRowChanged.connect(self.show_object)
        right.addWidget(self.objects)
        self.editor = QWidget()
        form = QFormLayout(self.editor)
        self.species = QComboBox()
        self.species.addItems(SPECIES)
        self.kind = QComboBox()
        self.kind.addItems(KINDS)
        form.addRow("Species", self.species)
        form.addRow("Object", self.kind)
        self.coords = []
        for name in ("Left", "Top", "Right (exclusive)", "Bottom (exclusive)"):
            spin = QSpinBox()
            spin.setRange(0, 1000000)
            form.addRow(name, spin)
            self.coords.append(spin)
        self.uncertain = QCheckBox()
        self.uncertain.setAccessibleName("Uncertain extent or mixed group")
        self.truncated = QCheckBox()
        self.truncated.setAccessibleName("Clipped by image edge")
        self.object_notes = QLineEdit()
        form.addRow("Uncertain extent / mixed group", self.uncertain)
        form.addRow("Clipped by image edge", self.truncated)
        form.addRow("Object notes", self.object_notes)
        right.addWidget(self.editor)
        self.apply_button = QPushButton("Apply object edits")
        self.apply_button.clicked.connect(self.apply_object)
        right.addWidget(self.apply_button)
        self.delete_button = QPushButton("Delete selected box")
        self.delete_button.clicked.connect(self.delete_box)
        right.addWidget(self.delete_button)
        review_form = QFormLayout()
        self.review_status = QComboBox()
        self.review_status.addItems(STATUSES)
        self.partition = QComboBox()
        self.partition.addItems(PARTITIONS)
        self.empty = QCheckBox()
        self.empty.setAccessibleName("No algae visible; confirm empty image")
        self.notes = QLineEdit()
        review_form.addRow("Image review", self.review_status)
        review_form.addRow("Partition", self.partition)
        review_form.addRow("No algae visible", self.empty)
        review_form.addRow("Image notes", self.notes)
        right.addLayout(review_form)
        for form, fields in ((form, (self.species, self.kind, *self.coords, self.uncertain, self.truncated, self.object_notes)),
                             (review_form, (self.review_status, self.partition, self.empty, self.notes))):
            for field in fields:
                label = form.labelForField(field)
                if label:
                    label.setStyleSheet("QLabel { color: #263650; } QLabel:disabled { color: #53657d; }")
        complete = QPushButton("Mark reviewed + next")
        complete.setObjectName("primaryButton")
        complete.clicked.connect(self.review_next)
        right.addWidget(complete)
        right.addStretch()
        splitter.addWidget(panel)
        splitter.setSizes([260, 780, 340])
        for widget in (self.species, self.kind):
            widget.currentIndexChanged.connect(self.object_changed)
        for widget in self.coords:
            widget.valueChanged.connect(self.object_changed)
        for widget in (self.uncertain, self.truncated):
            widget.toggled.connect(self.object_changed)
        self.object_notes.textEdited.connect(self.object_changed)
        self.review_status.currentIndexChanged.connect(self.mark_dirty)
        self.partition.currentIndexChanged.connect(self.mark_dirty)
        self.empty.toggled.connect(self.mark_dirty)
        self.notes.textEdited.connect(self.mark_dirty)
        self.load()

    def action(self, toolbar, text, shortcut, callback):
        action = QAction(text, self)
        action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback)
        toolbar.addAction(action)

    def set_drawing(self, enabled):
        self.viewer.cancel_drawing()
        self.viewer.drawing = enabled
        self.viewer.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def mark_dirty(self, *_):
        if not self.updating:
            self.dirty = True
            self.statusBar().showMessage("Unsaved edits. Ctrl+S saves this image.")

    def object_changed(self, *_):
        if not self.updating:
            self.object_edits = True
            self.mark_dirty()

    def refresh_lists(self):
        self.images.blockSignals(True)
        self.images.clear()
        for row in self.session.data["images"]:
            self.images.addItem(f"{row['status']}  |  {row['image_id']}")
            self.images.item(self.images.count() - 1).setToolTip(row["image_id"])
        self.images.setCurrentRow(self.index)
        self.images.blockSignals(False)
        reviewed = sum(r["status"] == "reviewed" for r in self.session.data["images"])
        self.progress.setText(f"{reviewed} / {len(self.session.data['images'])} images reviewed")

    def load(self):
        self.updating = True
        self.viewer.cancel_drawing()
        self.row = deepcopy(self.session.data["images"][self.index])
        source = self.session.verify_source(self.row)
        rgb = cv2.cvtColor(read_image(source), cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.viewer.set_image(image)
        self.review_status.setCurrentText(self.row["status"])
        self.partition.setCurrentText(self.row["partition"])
        self.empty.setChecked(self.row["empty_confirmed"])
        self.notes.setText(self.row["notes"])
        self.title.setText(f"{self.index + 1}/{len(self.session.data['images'])}  {self.row['image_id']}\n"
                           f"{self.row['width']} × {self.row['height']} source pixels · label each organism independently")
        self.refresh_lists()
        self.refresh_objects()
        self.dirty = False
        self.updating = False
        self.statusBar().showMessage(f"Saved annotations: {self.session.path}")

    def refresh_objects(self, selected=-1):
        self.objects.blockSignals(True)
        self.objects.clear()
        for number, obj in enumerate(self.row["objects"], 1):
            suffix = " · uncertain" if obj["uncertain"] else ""
            self.objects.addItem(f"{number}. {obj['species']} · {obj['kind']}{suffix}")
        self.objects.setCurrentRow(selected)
        self.objects.blockSignals(False)
        self.show_object(selected)

    def show_object(self, index):
        if getattr(self, "object_edits", False) and not self.apply_object():
            self.objects.blockSignals(True)
            self.objects.setCurrentRow(self.edit_index)
            self.objects.blockSignals(False)
            return
        previous_updating = self.updating
        self.updating = True
        self.edit_index = index
        enabled = 0 <= index < len(self.row["objects"])
        for widget in (self.apply_button, self.delete_button, self.species, self.kind,
                       self.uncertain, self.truncated, self.object_notes, *self.coords):
            widget.setEnabled(enabled)
        selected = None
        if enabled:
            obj = self.row["objects"][index]
            selected = obj["object_id"]
            self.species.setCurrentText(obj["species"])
            self.kind.setCurrentText(obj["kind"])
            for spin, value in zip(self.coords, obj["box"]):
                spin.setValue(value)
            self.uncertain.setChecked(obj["uncertain"])
            self.truncated.setChecked(obj["truncated"])
            self.object_notes.setText(obj["notes"])
        self.object_edits = False
        self.viewer.refresh_boxes(self.row["objects"], selected)
        self.updating = previous_updating

    def select_box(self, identifier):
        index = next(i for i, obj in enumerate(self.row["objects"]) if obj["object_id"] == identifier)
        self.objects.setCurrentRow(index)

    def add_box(self, box):
        if not self.apply_object():
            return
        self.row["objects"].append({"object_id": uuid4().hex, "box": box, "species": "unknown",
                                    "kind": "cell", "uncertain": False,
                                    "truncated": box[0] == 0 or box[1] == 0 or box[2] == self.row["width"] or box[3] == self.row["height"], "notes": ""})
        self.empty.setChecked(False)
        self.review_status.setCurrentText("pending")
        self.refresh_objects(len(self.row["objects"]) - 1)
        self.mark_dirty()

    def apply_object(self, *_):
        if not getattr(self, "object_edits", False):
            return True
        obj = deepcopy(self.row["objects"][self.edit_index])
        obj.update(box=[s.value() for s in self.coords], species=self.species.currentText(),
                   kind=self.kind.currentText(), uncertain=self.uncertain.isChecked() or self.kind.currentText() == "unresolved_group",
                   truncated=self.truncated.isChecked(), notes=self.object_notes.text())
        left, top, right, bottom = obj["box"]
        if not (0 <= left < right <= self.row["width"] and 0 <= top < bottom <= self.row["height"]):
            self.statusBar().showMessage("Invalid box: left < right and top < bottom must stay inside the image.")
            return False
        self.row["objects"][self.edit_index] = obj
        self.object_edits = False
        self.review_status.setCurrentText("pending")
        self.refresh_objects(self.edit_index)
        return True

    def delete_box(self):
        index = self.objects.currentRow()
        if index >= 0:
            self.object_edits = False
            del self.row["objects"][index]
            self.review_status.setCurrentText("pending")
            self.refresh_objects()
            self.mark_dirty()

    def save(self, *_):
        if not self.apply_object():
            return False
        self.row.update(status=self.review_status.currentText(), partition=self.partition.currentText(),
                        empty_confirmed=self.empty.isChecked(), notes=self.notes.text())
        try:
            self.session.save(self.index, self.row)
        except (ValueError, OSError) as error:
            self.statusBar().showMessage(f"Save failed: {error}")
            return False
        self.dirty = False
        self.refresh_lists()
        self.statusBar().showMessage("Image saved. Annotations persist when you close this window.")
        return True

    def can_leave(self):
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, "Unsaved image edits", "Save changes to this image before leaving?",
                                      QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                      QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def navigate(self, index):
        if index == self.index or not 0 <= index < len(self.session.data["images"]):
            return
        try:
            self.session.verify_source(self.session.data["images"][index])
        except (ValueError, OSError) as error:
            self.statusBar().showMessage(f"Cannot open image: {error}")
            allowed = False
        else:
            allowed = self.can_leave()
        if not allowed:
            self.images.blockSignals(True)
            self.images.setCurrentRow(self.index)
            self.images.blockSignals(False)
            return
        self.object_edits = False
        self.index = index
        self.load()

    def review_next(self):
        if not self.apply_object():
            return
        self.review_status.setCurrentText("reviewed")
        if self.save():
            if self.index + 1 < len(self.session.data["images"]):
                self.navigate(self.index + 1)
            else:
                self.statusBar().showMessage("Last image saved. Check pending images in the list, then export a snapshot.")

    def export(self):
        if self.dirty and not self.save():
            return
        parent = QFileDialog.getExistingDirectory(self, "Choose parent folder for a new export")
        if not parent:
            return
        from datetime import datetime
        destination = Path(parent) / f"detection_export_{datetime.now():%Y%m%d_%H%M%S}"
        try:
            self.session.export(destination)
        except (ValueError, OSError) as error:
            self.statusBar().showMessage(f"Export failed: {error}")
            return
        self.statusBar().showMessage(f"Snapshot exported: {destination}. Pending/uncertain regions have ignore=True.")

    def closeEvent(self, event):
        if self.can_leave():
            event.accept()
        else:
            event.ignore()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path, help="Create a new session")
    mode.add_argument("--session", type=Path, help="Resume an existing session")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--per-class", type=int, default=10, help="Sample this many files per source folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-list", type=Path, help="Optional text file of root-relative image IDs, one per line")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--export-dir", type=Path, help="Export a fresh snapshot without opening a window")
    args = parser.parse_args(argv)
    try:
        if args.session and args.image_list:
            raise ValueError("--image-list is only valid when creating a new session")
        identifiers = [s.strip() for s in args.image_list.read_text(encoding="utf-8-sig").splitlines() if s.strip()] if args.image_list else None
        session = DetectionSession(args.session) if args.session else DetectionSession.create(
            args.dataset_dir, args.output_dir, per_class=args.per_class, seed=args.seed, image_ids=identifiers)
        if args.export_dir:
            session.export(args.export_dir)
        if args.prepare_only or args.export_dir:
            print(f"Validated {len(session.data['images'])} images: {session.path}")
            return 0
        app = QApplication.instance() or QApplication(sys.argv[:1])
        configure_app(app)
        window = DetectionWindow(session)
        window.show()
        return app.exec()
    except Exception as error:
        print(f"Detection annotation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
