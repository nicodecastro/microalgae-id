"""Generate a synthetic microscopy-style demo without external image assets."""

from random import Random

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen

from microalgae.domain.results import Detection, ImageAnalysis


DEMO_IMAGE_ID = "demo-image"


def _draw_cell(painter: QPainter, x: int, y: int, length: int, angle: int) -> None:
    painter.save()
    painter.translate(x, y)
    painter.rotate(angle)
    painter.setPen(QPen(QColor(69, 82, 76, 200), 3))
    painter.setBrush(QColor(139, 151, 127, 180))
    painter.drawEllipse(QRectF(-length / 2, -24, length, 48))
    painter.setPen(QPen(QColor(66, 76, 65, 150), 2))
    painter.drawLine(-length // 2 + 16, 0, length // 2 - 16, 0)
    for offset in range(-length // 2 + 28, length // 2 - 12, 20):
        painter.drawLine(offset, -13, offset + 6, 13)
    painter.restore()


def create_demo() -> tuple[QImage, ImageAnalysis]:
    image = QImage(960, 620, QImage.Format.Format_RGB32)
    image.fill(QColor("#d7ddda"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    background = QLinearGradient(0, 0, image.width(), image.height())
    background.setColorAt(0, QColor("#c9d2d1"))
    background.setColorAt(1, QColor("#e0e4df"))
    painter.fillRect(image.rect(), background)

    random = Random(17)
    painter.setPen(Qt.PenStyle.NoPen)
    for _ in range(115):
        diameter = random.randint(3, 15)
        painter.setBrush(QColor(93, 113, 109, random.randint(13, 38)))
        painter.drawEllipse(
            QRectF(random.randint(0, 960), random.randint(0, 620), diameter, diameter)
        )

    _draw_cell(painter, 365, 300, 180, -20)
    _draw_cell(painter, 705, 420, 160, -25)

    painter.setBrush(QColor(15, 23, 42, 180))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(QRectF(22, 20, 160, 38), 8, 8)
    painter.setPen(QColor("#ffffff"))
    painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
    painter.drawText(QRectF(32, 23, 140, 30), Qt.AlignmentFlag.AlignCenter, "DEMO IMAGE")
    painter.end()

    analysis = ImageAnalysis(
        image_id=DEMO_IMAGE_ID,
        source="demo",
        detections=(
            Detection("demo-cell-1", 262, 240, 206, 118, "Nitzschia palea", 0.94),
            Detection("demo-cell-2", 615, 365, 180, 112, "Nitzschia palea", 0.91),
        ),
    )
    return image, analysis
