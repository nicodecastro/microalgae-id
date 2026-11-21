"""Desktop editor for persistent crop batches."""
from math import ceil, floor
from pathlib import Path
from PySide6.QtCore import Qt, QRectF, Signal, QEvent, QObject, QThreadPool, Slot
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QListWidget, QListWidgetItem, QLabel, QPushButton, QFileDialog, QMessageBox, QComboBox, QInputDialog, QProgressDialog, QToolButton
from microalgae.imaging.crop_review import ARTIFACT, CropBatch, DECISIONS, TARGET_CLASSES
from .image_viewer import ImageViewer


class ReviewTask(QObject):
    finished = Signal(object)

    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.result = None

    def run(self):
        try:
            self.result = self.operation()
        except Exception as error:
            self.finished.emit(str(error))
        else:
            self.finished.emit(None)


class ReviewProgress(QProgressDialog):
    def reject(self):
        pass

    def closeEvent(self, event):
        event.ignore()


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
    prediction_changed = Signal(object)

    def __init__(self, batch, parent=None, *, models=None, initial_image=0):
        super().__init__(parent)
        self.batch = batch
        self.models = models
        self.prediction_run = batch.data['collection'] == 'prediction'
        evaluation = batch.data['collection'] == 'evaluation'
        bar = self.menuBar()
        bar.setNativeMenuBar(False)
        self.import_folder_button = QToolButton(bar)
        self.import_folder_button.setText('Import image class folder')
        self.import_folder_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.import_folder_button.clicked.connect(self.open_class_folder)
        bar.setCornerWidget(self.import_folder_button, Qt.Corner.TopLeftCorner)
        self.import_folder_button.setVisible(not self.prediction_run)
        if models is not None:
            models.add_menu(self)
        self.row = None
        self.crop = None
        self._task = None
        self._progress = None
        self.review_counter = QLabel()
        self.review_counter.setObjectName('summaryCard')
        self.review_counter.setMinimumHeight(36)
        self.species_counter = QLabel()
        self.species_counter.setObjectName('mutedText')
        self.review_counter.setVisible(not self.prediction_run)
        self.species_counter.setVisible(not self.prediction_run)
        self.image_status = QLabel()
        self.image_status.setObjectName('summaryCard')
        self.image_status.setWordWrap(True)
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
        image_panel.addWidget(self.image_status)
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
        if models is not None:
            self.classification = QLabel()
            self.classification.setObjectName('summaryCard')
            self.classification.setTextFormat(Qt.TextFormat.PlainText)
            self.classification.setWordWrap(True)
            side.addWidget(self.classification)
        self.crops = QListWidget()
        self.crops.setToolTip('Caps Lock: next crop (wraps to first)')
        side.addWidget(self.crops)
        QApplication.instance().installEventFilter(self)
        self.preview = QLabel()
        self.preview.setMinimumSize(240, 200)
        side.addWidget(self.preview)
        self.target = QComboBox()
        self.target.addItems(['unknown', *TARGET_CLASSES])
        side.addWidget(self.target)
        self.target.setVisible(not evaluation and not self.prediction_run)
        self.target.currentTextChanged.connect(self.change_target)
        actions = QGridLayout()
        actions.setDefaultPositioning(2, Qt.Orientation.Horizontal)
        actions.setColumnStretch(0, 1)
        actions.setColumnStretch(1, 1)
        side.addLayout(actions)
        decisions = () if self.prediction_run else (*DECISIONS[1:4], ARTIFACT) if evaluation else DECISIONS[1:]
        for decision, key in zip(decisions, ['A', 'N', 'U', 'R']):
            label = {'Accept target': 'Eligible target', 'Unsure / mixed': 'Needs review'}.get(decision, decision) if evaluation else decision
            button = QPushButton(f'{label} ({key})')
            actions.addWidget(button)
            button.clicked.connect(lambda checked=False, d=decision: self.decide(d))
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(button.click)
        for text, key, callback in [('Draw missed target', 'D', lambda: self.draw(False)), ('Redraw selected box', 'B', lambda: self.draw(True)), ('Mark image reviewed', 'M', self.complete), ('Export training snapshot', None, self.export)]:
            if (evaluation and key is None) or (self.prediction_run and key in ('M', None)):
                continue
            button = QPushButton(f'{text} ({key})' if key else text)
            actions.addWidget(button)
            button.clicked.connect(callback)
            if key:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(button.click)
        if models is not None:
            self.classify_button = QPushButton('Rerun edited crops' if self.prediction_run else 'Predict species')
            self.classify_button.clicked.connect(self.rerun_prediction_edits if self.prediction_run else self.classify_crop)
            actions.addWidget(self.classify_button)
            self.show_classification()
        if self.prediction_run:
            self.remove_prediction_button = QPushButton('Remove false detection')
            self.remove_prediction_button.setEnabled(False)
            self.remove_prediction_button.clicked.connect(self.remove_prediction_crop)
            actions.addWidget(self.remove_prediction_button)
        self.images.currentRowChanged.connect(self.show_image)
        self.crops.currentRowChanged.connect(self.show_crop)
        self.viewer.box_drawn.connect(self.add_box)
        self.viewer.detection_selected.connect(self.select_box)
        for row in batch.data['images']:
            self.add_image_item(row)
        self.images.setCurrentRow(initial_image)

    def add_image_item(self, row):
        identifier = row['image_id']
        legacy_import = row.get('batch_local_source') and 'source_path' not in row
        item = QListWidgetItem(row.get('title', Path(identifier).name if legacy_import else identifier))
        item.setToolTip(identifier)
        self.images.addItem(item)

    def open_class_folder(self):
        if self._task is not None:
            return
        folder = QFileDialog.getExistingDirectory(self, 'Import image class folder', self.batch.data['root'])
        if folder:
            self.start_task('Importing image class folder…', lambda: self.batch.import_class_folder(folder),
                            self.finish_import)

    @Slot(object)
    def finish_import(self, error):
        result = self._task.result
        self.clear_task()
        first_new = self.images.count()
        new_rows = self.batch.data['images'][first_new:]
        for row in new_rows:
            self.add_image_item(row)
        if new_rows:
            self.viewer.drawing = False
            self.images.setCurrentRow(first_new)
            self.update_review_counter()
        if error is not None:
            QMessageBox.warning(self, 'Could not import image class folder', error)
            return
        duplicates, failures = result
        summary = f'Imported {len(new_rows)} · Skipped {duplicates} existing name(s) · Failed {len(failures)}'
        self.statusBar().showMessage(summary, 8000)
        if failures:
            QMessageBox.warning(self, 'Some images could not be imported', '\n'.join([summary, '', *failures]))

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
            return False
        self.update_review_counter()
        if self.prediction_run:
            self.prediction_changed.emit(self.row)
        return True

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
        self.update_image_status()

    def update_image_status(self):
        if self.row is None:
            self.image_status.setText('No image selected')
            return
        status = 'Reviewed' if self.row['reviewed'] else self.row['status'].capitalize()
        if self.prediction_run and status == 'Zero proposals' and any(c['active'] for c in self.row['crops']):
            status = 'Edited crops'
        self.image_status.setText(f'Current image · {status}')

    def change_target(self, text):
        if self.row and self.row['target'] != text:
            self.row['target'] = text
            self.row['reviewed'] = False
            self.safely(lambda: None)

    def show_image(self, index):
        if index < 0:
            return
        preserve_view = self.row is self.batch.data['images'][index]
        self.row = self.batch.data['images'][index]
        self.update_image_status()
        self.crop = None
        self.target.blockSignals(True)
        self.target.setCurrentText(self.row['target'] if self.target.findText(self.row['target']) >= 0 else 'unknown')
        self.target.blockSignals(False)
        try:
            from microalgae.domain.results import Detection, ImageAnalysis
            image = self.batch.source(self.row)
            rgb = image[:, :, ::-1].copy()
            qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
            detections = tuple((Detection(c['crop_id'], c['box'][0], c['box'][1], c['box'][2] - c['box'][0], c['box'][3] - c['box'][1], 'Proposed region') for c in self.row['crops'] if c['active']))
            self.viewer.set_image(qimage, ImageAnalysis(self.row['image_id'], detections, 'model'), preserve_view=preserve_view)
            self.statusBar().showMessage('Original pixel orientation · ' + ('Reviewed' if self.row['reviewed'] else self.row['status']))
        except Exception as error:
            self.viewer.clear_image()
            self.statusBar().showMessage(str(error))
        self.crops.clear()
        self.active = [c for c in self.row['crops'] if c['active']]
        for c in self.active:
            self.crops.addItem(self.crop_label(c))
        self.crops.setCurrentRow(0)
        if not self.active:
            self.preview.clear()
            self.show_crop(-1)

    def show_crop(self, index):
        self.crop = self.active[index] if 0 <= index < len(self.active) else None
        if self.crop:
            self.preview.setPixmap(QPixmap(str(self.batch.directory / self.crop['path'])).scaled(350, 300, Qt.AspectRatioMode.KeepAspectRatio))
            self.viewer.select_detection(self.crop['crop_id'])
        self.show_classification()
        if self.prediction_run:
            self.remove_prediction_button.setEnabled(self.crop is not None)

    @staticmethod
    def crop_label(crop):
        parts = [crop['decision'], crop['crop_id'][:8]]
        result = crop.get('classification')
        if result:
            parts.append(result.get('species') or 'Prediction failed')
        elif crop.get('suggestion'):
            parts.append('Suggested ' + crop['suggestion'])
        return ' · '.join(parts)

    def show_classification(self):
        if self.models is None:
            return
        eligible = self.crop is not None and self.crop['decision'] == 'Accept target'
        self.classify_button.setEnabled(self.has_prediction_edits() if self.prediction_run else eligible)
        result = self.crop.get('classification') if self.crop and (eligible or self.prediction_run) else None
        if result:
            outcome = f"Prediction: {result['species']}" if result.get('species') else f"Prediction failed: {result['error']}"
            self.classification.setText(f"{outcome}\nModel: {result['model']}")
        elif self.crop is not None and self.prediction_run:
            self.classification.setText('Needs prediction')
        elif self.crop is not None and self.crop['decision'] == 'pending':
            self.classification.setText('Review this crop before predicting its species.')
        else:
            self.classification.clear()
        self.classification.setVisible(bool(self.classification.text()))

    def has_prediction_edits(self):
        from microalgae.imaging.prediction import edited_crops

        return next(edited_crops(self.batch), None) is not None

    def remove_prediction_crop(self):
        if self.crop is not None and self.safely(lambda: self.crop.update(active=False, decision=ARTIFACT)):
            self.show_image(self.images.currentRow())

    def rerun_prediction_edits(self):
        from microalgae.imaging.prediction import rerun_edits

        if self._task is not None or not self.has_prediction_edits():
            return
        try:
            classifier = self.models.original_classifier(self, self.batch.data['classifier'])
        except Exception as error:
            QMessageBox.warning(self, 'Could not rerun predictions', str(error))
            return
        if classifier is not None:
            self.start_task('Predicting edited crops…', lambda: rerun_edits(self.batch, classifier),
                            self.finish_prediction_edits)

    @Slot(object)
    def finish_prediction_edits(self, error):
        self.clear_task()
        if error is not None:
            QMessageBox.warning(self, 'Could not finish predictions', error)
            self.batch = CropBatch(self.batch.directory)
        for row in self.batch.data['images']:
            self.prediction_changed.emit(row)
        selected = self.crops.currentRow()
        self.show_image(self.images.currentRow())
        if 0 <= selected < self.crops.count():
            self.crops.setCurrentRow(selected)

    def classify_crop(self):
        if self._task is not None or not self.classify_button.isEnabled():
            return
        try:
            classifier = self.models.get_classifier(self)
        except Exception as error:
            QMessageBox.warning(self, 'Could not predict species', str(error))
            return
        if classifier is not None:
            crop = self.crop
            self.start_task('Predicting species…', lambda: self.batch.classify(crop, classifier),
                            self.finish_classification)

    @Slot(object)
    def finish_classification(self, error):
        self.clear_task()
        if error is not None:
            QMessageBox.warning(self, 'Could not save prediction', error)
        if self.crop is not None:
            self.crops.currentItem().setText(self.crop_label(self.crop))
        self.show_classification()

    def select_box(self, identifier):
        for index, crop in enumerate(self.active):
            if crop['crop_id'] == identifier:
                self.crops.setCurrentRow(index)

    def decide(self, decision):
        if self.crop:
            if decision == 'Discard crop':
                self.start_discard()
                return
            self.safely(lambda: self.batch.decide(self.row, self.crop, decision))
            self.crops.currentItem().setText(self.crop_label(self.crop))
            self.show_classification()

    def start_discard(self):
        if self._task is not None:
            return
        row, crop = self.row, self.crop
        self.start_task('Discarding crop…', lambda: self.batch.discard(row, crop), self.finish_discard)

    def start_task(self, message, operation, finish):
        self._progress = ReviewProgress(message, '', 0, 0, self)
        self._progress.setCancelButton(None)
        self._progress.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self._progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._progress.show()
        self._task = ReviewTask(operation)
        self._task.finished.connect(finish)
        QThreadPool.globalInstance().start(self._task.run)

    def clear_task(self):
        self._progress.hide()
        self._progress.deleteLater()
        self._progress = None
        self._task = None

    @Slot(object)
    def finish_discard(self, error):
        self.clear_task()
        index = self.crops.currentRow()
        self.update_review_counter()
        self.show_image(self.images.currentRow())
        if self.active:
            self.crops.setCurrentRow(min(index, len(self.active) - 1))
        if error is not None:
            QMessageBox.warning(self, 'Crop review', error)

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
            if Path(name).name != name:
                return
            self.safely(lambda: self.batch.export(Path(parent) / name))
