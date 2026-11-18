"""Desktop editor for persistent crop batches."""
from math import ceil, floor
from PySide6.QtCore import Qt, QRectF, Signal, QEvent
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel, QPushButton, QFileDialog, QMessageBox, QComboBox, QInputDialog
from microalgae.imaging.crop_review import CropBatch, DECISIONS
from .image_viewer import ImageViewer

class CropViewer(ImageViewer):
    box_drawn = Signal(list)

    def __init__(self):
        super().__init__()
        self.drawing = False
        self.start = None
        self.preview = None

    def mousePressEvent(self, event):
        if self.drawing and event.button() == Qt.MouseButton.LeftButton:
            self.start = self.mapToScene(event.position().toPoint())
            self.preview = self.scene().addRect(QRectF(self.start, self.start))
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.preview.setRect(QRectF(self.start, self.mapToScene(event.position().toPoint())).normalized().intersected(self.sceneRect()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.start is not None:
            rect = self.preview.rect()
            self.scene().removeItem(self.preview)
            self.start = None
            if rect.width() >= 1 and rect.height() >= 1:
                self.box_drawn.emit([floor(rect.left()), floor(rect.top()), ceil(rect.right()), ceil(rect.bottom())])
            return
        super().mouseReleaseEvent(event)

class CropReviewWindow(QMainWindow):

    def __init__(self, batch, parent=None):
        super().__init__(parent)
        self.batch = batch
        self.row = None
        self.crop = None
        self.review_counter = QLabel()
        self.review_counter.setObjectName('summaryCard')
        self.review_counter.setMinimumHeight(36)
        self.species_counter = QLabel()
        self.species_counter.setObjectName('mutedText')
        self.update_review_counter()
        self.setWindowTitle('Crop review')
        self.resize(1300, 850)
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QHBoxLayout(widget)
        image_panel = QVBoxLayout()
        layout.addLayout(image_panel, 1)
        image_panel.addWidget(self.review_counter)
        image_panel.addWidget(self.species_counter)
        self.images = QListWidget()
        image_panel.addWidget(self.images)
        for key, step in [('PgUp', -1), ('PgDown', 1)]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda step=step: self.navigate_image(step))
        self.images.setToolTip('Page Up: previous image\nPage Down: next image')
        middle = QVBoxLayout()
        layout.addLayout(middle, 3)
        self.viewer = CropViewer()
        middle.addWidget(self.viewer)
        side = QVBoxLayout()
        layout.addLayout(side, 2)
        self.crops = QListWidget()
        self.crops.setToolTip('Caps Lock: next crop (wraps to first)')
        side.addWidget(self.crops)
        QApplication.instance().installEventFilter(self)
        self.preview = QLabel()
        self.preview.setMinimumSize(240, 200)
        side.addWidget(self.preview)
        self.target = QComboBox()
        self.target.addItems(['unknown', 'm_aeruginosa', 'n_fruticosa', 'n_palea'])
        side.addWidget(self.target)
        self.target.currentTextChanged.connect(self.change_target)
        for decision, key in zip(DECISIONS[1:], ['A', 'N', 'U', 'R']):
            button = QPushButton(f'{decision} ({key})')
            side.addWidget(button)
            button.clicked.connect(lambda checked=False, d=decision: self.decide(d))
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(button.click)
        for text, key, callback in [('Draw missed target', 'D', lambda: self.draw(False)), ('Redraw selected box', 'B', lambda: self.draw(True)), ('Mark image reviewed', 'M', self.complete), ('Export training snapshot', None, self.export)]:
            button = QPushButton(f'{text} ({key})' if key else text)
            side.addWidget(button)
            button.clicked.connect(callback)
            if key:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(button.click)
        self.images.currentRowChanged.connect(self.show_image)
        self.crops.currentRowChanged.connect(self.show_crop)
        self.viewer.box_drawn.connect(self.add_box)
        self.viewer.detection_selected.connect(self.select_box)
        for row in batch.data['images']:
            self.images.addItem(row['image_id'])
        self.images.setCurrentRow(0)

    def navigate_image(self, step):
        index = self.images.currentRow() + step
        if 0 <= index < self.images.count():
            self.images.setCurrentRow(index)

    def next_crop(self):
        count = self.crops.count()
        if count:
            self.crops.setCurrentRow((self.crops.currentRow() + 1) % count)

    def eventFilter(self, watched, event):
        if (event.type() == QEvent.Type.KeyPress
                and event.key() == Qt.Key.Key_CapsLock
                and isinstance(watched, QWidget)
                and watched.window() == self):
            if not event.isAutoRepeat():
                self.next_crop()
            return True
        return super().eventFilter(watched, event)

    def safely(self, callback):
        try:
            callback()
            self.batch.save()
        except Exception as error:
            QMessageBox.warning(self, 'Crop review', str(error))
        self.update_review_counter()

    def update_review_counter(self):
        images = self.batch.data['images']
        reviewed = sum(row['reviewed'] for row in images)
        self.review_counter.setText(f'{reviewed}/{len(images)} reviewed')
        counts = []
        for species, label in [('m_aeruginosa', 'M. aeruginosa'), ('n_fruticosa', 'N. fruticosa'), ('n_palea', 'N. palea')]:
            sources = [row for row in images if row['target'] == species]
            completed = sum(row['reviewed'] for row in sources)
            counts.append(f'{label}: {completed}/{len(sources)} reviewed')
        self.species_counter.setText('\n'.join(counts))

    def change_target(self, text):
        if self.row and self.row['target'] != text:
            self.row['target'] = text
            self.row['reviewed'] = False
            self.safely(lambda: None)

    def show_image(self, index):
        if index < 0:
            return
        self.row = self.batch.data['images'][index]
        self.crop = None
        self.target.setCurrentText(self.row['target'] if self.row['target'] in ['m_aeruginosa', 'n_fruticosa', 'n_palea'] else 'unknown')
        try:
            from microalgae.domain.results import Detection, ImageAnalysis
            image = self.batch.source(self.row)
            rgb = image[:, :, ::-1].copy()
            qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
            detections = tuple((Detection(c['crop_id'], c['box'][0], c['box'][1], c['box'][2] - c['box'][0], c['box'][3] - c['box'][1], 'Proposed region') for c in self.row['crops'] if c['active']))
            self.viewer.set_image(qimage, ImageAnalysis(self.row['image_id'], detections, 'model'))
            self.statusBar().showMessage('Original pixel orientation · ' + ('Reviewed' if self.row['reviewed'] else self.row['status']))
        except Exception as error:
            self.viewer.clear_image()
            self.statusBar().showMessage(str(error))
        self.crops.clear()
        self.active = [c for c in self.row['crops'] if c['active']]
        for c in self.active:
            self.crops.addItem(c['decision'] + ' · ' + c['crop_id'][:8] + (' · Suggested ' + c['suggestion'] if c.get('suggestion') else ''))
        self.crops.setCurrentRow(0)
        if not self.active:
            self.preview.clear()

    def show_crop(self, index):
        self.crop = self.active[index] if 0 <= index < len(self.active) else None
        if self.crop:
            self.preview.setPixmap(QPixmap(str(self.batch.directory / self.crop['path'])).scaled(350, 300, Qt.AspectRatioMode.KeepAspectRatio))
            self.viewer.select_detection(self.crop['crop_id'])

    def select_box(self, identifier):
        for index, crop in enumerate(self.active):
            if crop['crop_id'] == identifier:
                self.crops.setCurrentRow(index)

    def decide(self, decision):
        if self.crop:
            self.safely(lambda: self.batch.decide(self.row, self.crop, decision))
            self.crops.currentItem().setText(decision + ' · ' + self.crop['crop_id'][:8])

    def draw(self, replace):
        self.replaces = self.crop['crop_id'] if replace and self.crop else None
        self.viewer.drawing = True
        self.statusBar().showMessage('Drag a tight target box. Add separate boxes to replace a merged region.')

    def add_box(self, box):
        self.safely(lambda: self.batch.add(self.row, box, replaces=self.replaces))
        self.viewer.drawing = False
        self.show_image(self.images.currentRow())
        self.crops.setCurrentRow(len(self.active) - 1)

    def complete(self):
        self.safely(lambda: self.batch.complete(self.row))
        self.show_image(self.images.currentRow())

    def export(self):
        parent = QFileDialog.getExistingDirectory(self, 'Choose parent for a new export')
        if not parent:
            return
        name, ok = QInputDialog.getText(self, 'Export', 'New directory name')
        if ok and name:
            from pathlib import Path
            if Path(name).name != name:
                return
            self.safely(lambda: self.batch.export(Path(parent) / name))
