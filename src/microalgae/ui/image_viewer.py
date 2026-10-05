"""Zoomable image canvas with selectable, image-coordinate annotations."""

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from microalgae.domain.results import ImageAnalysis


class ImageViewer(QGraphicsView):
    detection_selected = Signal(str)
    zoom_changed = Signal(int)
    files_dropped = Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QColor("#101a30"))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._has_image = False
        self._fit_on_resize = True
        self._annotations_visible = True
        self._overlay_items: dict[str, QGraphicsRectItem] = {}
        self._selected_id: str | None = None
        self._pan_position: QPointF | None = None

    def set_image(self, image: QImage, analysis: ImageAnalysis | None = None) -> None:
        self._scene.clear()
        self._overlay_items.clear()
        self._selected_id = None
        self._has_image = not image.isNull()
        if not self._has_image:
            return

        pixmap_item = self._scene.addPixmap(QPixmap.fromImage(image))
        pixmap_item.setZValue(0)
        self._scene.setSceneRect(QRectF(0, 0, image.width(), image.height()))
        if analysis is not None:
            for number, detection in enumerate(analysis.detections, start=1):
                box = QGraphicsRectItem(QRectF(0, 0, detection.width, detection.height))
                box.setPos(detection.x, detection.y)
                box.setZValue(1)
                box.setData(0, detection.identifier)
                box.setCursor(Qt.CursorShape.PointingHandCursor)
                box.setBrush(QColor(59, 130, 246, 23))
                self._scene.addItem(box)

                marker = QGraphicsSimpleTextItem(f"#{number}", box)
                marker.setPos(3, -23)
                marker.setBrush(QColor("#ffffff"))
                marker.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
                marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                self._overlay_items[detection.identifier] = box

        self._update_overlay_style()
        self.set_annotations_visible(self._annotations_visible)
        self.fit_image()

    def clear_image(self) -> None:
        self._scene.clear()
        self._overlay_items.clear()
        self._selected_id = None
        self._has_image = False
        self._fit_on_resize = True
        self.resetTransform()

    def fit_image(self) -> None:
        if not self._has_image or self.viewport().width() <= 0:
            return
        self._fit_on_resize = True
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_changed.emit(round(self.transform().m11() * 100))

    def zoom(self, factor: float) -> None:
        if not self._has_image:
            return
        current = self.transform().m11()
        if not 0.02 <= current * factor <= 20:
            return
        self._fit_on_resize = False
        self.scale(factor, factor)
        self.zoom_changed.emit(round(self.transform().m11() * 100))

    def set_annotations_visible(self, visible: bool) -> None:
        self._annotations_visible = visible
        for box in self._overlay_items.values():
            box.setVisible(visible)

    def select_detection(self, identifier: str | None) -> None:
        self._selected_id = identifier if identifier in self._overlay_items else None
        self._update_overlay_style()

    def _update_overlay_style(self) -> None:
        for identifier, box in self._overlay_items.items():
            selected = identifier == self._selected_id
            pen = QPen(QColor("#f59e0b" if selected else "#3b82f6"), 3 if selected else 2)
            pen.setCosmetic(True)
            box.setPen(pen)
            box.setBrush(QColor(245, 158, 11, 35) if selected else QColor(59, 130, 246, 23))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_on_resize:
            self.fit_image()

    def wheelEvent(self, event) -> None:
        if self._has_image and event.angleDelta().y():
            self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._has_image:
            scene_position = self.mapToScene(event.position().toPoint())
            for identifier, box in self._overlay_items.items():
                if self._annotations_visible and box.sceneBoundingRect().contains(scene_position):
                    self.select_detection(identifier)
                    self.detection_selected.emit(identifier)
                    event.accept()
                    return
            self._pan_position = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_position is not None:
            delta = event.position() - self._pan_position
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(delta.y()))
            self._pan_position = event.position()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._pan_position is not None:
            self._pan_position = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(
            url.isLocalFile() for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(
            url.isLocalFile() for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()
