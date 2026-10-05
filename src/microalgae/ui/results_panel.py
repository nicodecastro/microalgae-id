"""Current-image result summary and selectable detection cards."""

from collections import Counter

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from microalgae.domain.results import ImageAnalysis


class ResultsPanel(QWidget):
    detection_selected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sidePanel")
        self.setMinimumWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Results")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.demo_badge = QLabel("DEMO DATA · sample results only")
        self.demo_badge.setObjectName("demoBadge")
        self.demo_badge.setWordWrap(True)
        self.demo_badge.hide()
        layout.addWidget(self.demo_badge)

        self.message = QLabel("Add an image or open the demo to begin.")
        self.message.setWordWrap(True)
        self.message.setObjectName("mutedText")
        layout.addWidget(self.message)

        self.summary = QLabel()
        self.summary.setObjectName("summaryCard")
        self.summary.setWordWrap(True)
        self.summary.hide()
        layout.addWidget(self.summary)

        self.distribution = QLabel()
        self.distribution.setObjectName("mutedText")
        self.distribution.setWordWrap(True)
        self.distribution.hide()
        layout.addWidget(self.distribution)

        self.result_list = QListWidget()
        self.result_list.setObjectName("resultList")
        self.result_list.currentItemChanged.connect(self._on_current_item_changed)
        self.result_list.hide()
        layout.addWidget(self.result_list, 1)
        layout.addStretch(1)
        self._layout = layout
        self._spacer_index = layout.count() - 1

    def set_analysis(self, analysis: ImageAnalysis | None, *, has_image: bool) -> None:
        self.result_list.clear()
        self.demo_badge.hide()
        self.summary.hide()
        self.distribution.hide()
        self.result_list.hide()

        if analysis is None:
            self._layout.setStretchFactor(self.result_list, 0)
            self._layout.setStretch(self._spacer_index, 1)
            self.message.setText(
                "Analysis is not available yet. This prototype lets you review images "
                "without classifying them."
                if has_image
                else "Add an image or open the demo to begin."
            )
            self.message.show()
            return

        is_demo = analysis.source == "demo"
        self._layout.setStretchFactor(self.result_list, 1)
        self._layout.setStretch(self._spacer_index, 0)
        self.demo_badge.setVisible(is_demo)
        self.message.hide()
        counts = Counter(detection.species for detection in analysis.detections)
        probabilities = [
            detection.probability
            for detection in analysis.detections
            if detection.probability is not None
            and (is_demo or detection.probability_validated)
        ]
        mean_text = (
            f"{sum(probabilities) / len(probabilities):.1%}"
            if probabilities and len(probabilities) == len(analysis.detections)
            else "—"
        )
        prefix = "Sample " if is_demo else ""
        self.summary.setText(
            f"{prefix}cells: {len(analysis.detections)}\n"
            f"{prefix}species: {len(counts)}\n"
            f"Mean {prefix.lower()}probability: {mean_text}"
        )
        self.summary.show()
        distribution = ", ".join(f"{species} ({count})" for species, count in counts.items())
        self.distribution.setText(f"Species distribution: {distribution or 'None'}")
        self.distribution.show()

        for number, detection in enumerate(analysis.detections, start=1):
            show_probability = detection.probability is not None and (
                is_demo or detection.probability_validated
            )
            score = (
                f"{detection.probability:.0%} {'demo ' if is_demo else ''}probability"
                if show_probability
                else "Probability unavailable"
            )
            item = QListWidgetItem(f"Cell #{number} · {detection.species}\n{score}")
            item.setData(Qt.ItemDataRole.UserRole, detection.identifier)
            item.setToolTip(
                f"x={detection.x:g}, y={detection.y:g}, "
                f"width={detection.width:g}, height={detection.height:g}"
            )
            item.setSizeHint(QSize(0, 64))
            self.result_list.addItem(item)
        self.result_list.show()

    def select_detection(self, identifier: str | None) -> None:
        previous = self.result_list.blockSignals(True)
        try:
            self.result_list.clearSelection()
            if identifier is None:
                self.result_list.setCurrentRow(-1)
            else:
                for row in range(self.result_list.count()):
                    item = self.result_list.item(row)
                    if item.data(Qt.ItemDataRole.UserRole) == identifier:
                        self.result_list.setCurrentItem(item)
                        self.result_list.scrollToItem(item)
                        break
        finally:
            self.result_list.blockSignals(previous)

    def _on_current_item_changed(self, current, _previous) -> None:
        if current is not None:
            identifier = current.data(Qt.ItemDataRole.UserRole)
            if identifier:
                self.detection_selected.emit(identifier)
