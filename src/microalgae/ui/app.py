"""PySide6 application entry point and local visual style."""

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from microalgae.ui.main_window import MainWindow


STYLE = """
QMainWindow, QWidget#centerPanel { background: #f5f7fb; color: #17233a; }
QWidget#sidePanel { background: #ffffff; color: #17233a; }
QWidget#panelDivider { background: #dde5f0; }
QToolBar#mainToolbar {
    background: #ffffff; border: none; border-bottom: 1px solid #dde5f0;
    spacing: 7px; padding: 7px 12px;
}
QToolBar QToolButton { color: #263650; padding: 8px 10px; border-radius: 7px; }
QToolBar QToolButton:hover { background: #eaf1ff; }
QToolBar QToolButton:checked { background: #dbeafe; color: #1555c0; }
QToolBar QToolButton:disabled { color: #a3afc2; }
QToolButton#imageNavigation {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
        stop: 0 rgba(242, 245, 249, 232), stop: 1 rgba(217, 225, 235, 224));
    border: 1px solid rgba(248, 251, 255, 232); border-radius: 24px;
}
QToolButton#imageNavigation:disabled {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
        stop: 0 rgba(156, 166, 181, 221), stop: 1 rgba(124, 135, 151, 215));
    border-color: rgba(170, 180, 194, 225);
}
QToolButton#imageNavigation:focus:enabled { border-color: #5c91f0; }
QToolButton#imageHeaderTool { background: transparent; border: 1px solid transparent; border-radius: 7px; }
QToolButton#imageHeaderTool:hover:enabled, QToolButton#imageHeaderTool:focus {
    background: #eaf1ff; border-color: #5c91f0;
}
QSplitter::handle { background: #dde5f0; width: 1px; }
QLabel#brandTitle { color: #10213b; font-size: 16px; font-weight: 600; }
QLabel#brandSubtitle { color: #66758d; font-size: 11px; }
QLabel#sectionTitle { color: #17233a; font-size: 17px; font-weight: 600; }
QLabel#emptyTitle { color: #eaf2ff; font-size: 23px; font-weight: 600; }
QLabel#mutedText { color: #66758d; font-size: 12px; }
QLabel#imageStatusMessage { color: #43546e; font-size: 12px; }
QProgressBar#imageStatusProgress { background: #dce6f5; border: none; border-radius: 1px; }
QProgressBar#imageStatusProgress::chunk { background: #2164e8; border-radius: 1px; }
QWidget#emptyPage { background: #0f172b; }
QWidget#emptyPage QLabel#mutedText { color: #b8c8df; font-size: 14px; }
QPushButton {
    background: #ffffff; color: #1b3558; border: 1px solid #b9c8dc;
    border-radius: 8px; padding: 9px 13px; min-height: 20px;
}
QPushButton:hover { background: #eef4ff; border-color: #5c91f0; }
QPushButton#primaryButton, QPushButton#predictCurrentButton {
    background: #2164e8; color: white; border-color: #2164e8;
}
QPushButton#primaryButton:hover, QPushButton#predictCurrentButton:hover {
    background: #1553c9; border-color: #1553c9;
}
QWidget#sidePanel QPushButton#primaryButton { border-radius: 6px; padding: 3px 9px; min-height: 16px; }
QWidget#predictionControls QPushButton { border-radius: 7px; padding: 6px 10px; min-height: 18px; }
QWidget#predictionControls QPushButton:disabled {
    background: #edf1f7; border-color: #dde5f0; color: #66758d;
}
QPushButton#imageAnnotationsButton {
    background: #2164e8; color: #ffffff; border-color: #2164e8;
    font-size: 12px; border-radius: 8px; padding: 6px 10px;
}
QPushButton#imageAnnotationsButton:hover:enabled, QPushButton#imageAnnotationsButton:focus {
    background: #1553c9; border-color: #1553c9;
}
QPushButton#imageAnnotationsButton:disabled { background: #e8edf4; border-color: #e8edf4; color: #66758d; }
QPushButton#dismissImageStatus { min-height: 0; padding: 0 7px; border-radius: 5px; font-size: 12px; }
QListWidget#imageList, QListWidget#resultList {
    background: #ffffff; color: #253550; border: 1px solid #dde5f0;
    border-radius: 9px; padding: 4px; outline: 0;
}
QListWidget::item { border-radius: 7px; padding: 9px; margin: 2px; }
QListWidget::item:hover { background: #edf3ff; }
QListWidget::item:selected { background: #dceaff; color: #17469b; }
QLabel#summaryCard {
    color: #173662; background: #edf4ff; border: 1px solid #c9ddff;
    border-radius: 9px; padding: 12px; line-height: 1.4;
}
QLabel#statusBadge { color: #326955; font-size: 11px; font-weight: 600; }
QStatusBar { background: #ffffff; color: #63728b; border-top: 1px solid #dde5f0; }
"""


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MicroalgaeID")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()
