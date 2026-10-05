"""PySide6 application entry point and local visual style."""

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from microalgae.ui.main_window import MainWindow


STYLE = """
QMainWindow, QWidget#centerPanel { background: #f5f7fb; color: #17233a; }
QWidget#sidePanel { background: #ffffff; color: #17233a; }
QToolBar#mainToolbar {
    background: #ffffff; border: none; border-bottom: 1px solid #dde5f0;
    spacing: 7px; padding: 7px 12px;
}
QToolBar QToolButton { color: #263650; padding: 8px 10px; border-radius: 7px; }
QToolBar QToolButton:hover { background: #eaf1ff; }
QToolBar QToolButton:checked { background: #dbeafe; color: #1555c0; }
QToolBar QToolButton:disabled { color: #a3afc2; }
QSplitter::handle { background: #dde5f0; width: 1px; }
QLabel#brandTitle { color: #10213b; font-size: 20px; font-weight: 700; }
QLabel#sectionTitle { color: #17233a; font-size: 17px; font-weight: 600; }
QLabel#emptyTitle { color: #eaf2ff; font-size: 23px; font-weight: 600; }
QLabel#mutedText { color: #66758d; font-size: 12px; }
QWidget#emptyPage { background: #101a30; }
QWidget#emptyPage QLabel#mutedText { color: #b8c8df; font-size: 14px; }
QPushButton {
    background: #ffffff; color: #1b3558; border: 1px solid #b9c8dc;
    border-radius: 8px; padding: 9px 13px; min-height: 20px;
}
QPushButton:hover { background: #eef4ff; border-color: #5c91f0; }
QPushButton#primaryButton { background: #2164e8; color: white; border-color: #2164e8; }
QPushButton#primaryButton:hover { background: #1553c9; border-color: #1553c9; }
QListWidget#imageList, QListWidget#resultList {
    background: #ffffff; color: #253550; border: 1px solid #dde5f0;
    border-radius: 9px; padding: 4px; outline: 0;
}
QListWidget::item { border-radius: 7px; padding: 9px; margin: 2px; }
QListWidget::item:hover { background: #edf3ff; }
QListWidget::item:selected { background: #dceaff; color: #17469b; }
QLabel#demoBadge {
    color: #8b4c00; background: #fff2d9; border: 1px solid #f4cb84;
    border-radius: 7px; padding: 8px; font-weight: 600;
}
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
