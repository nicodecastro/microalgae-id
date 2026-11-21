"""Shared model selections for the main window and crop review."""

from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QMessageBox


def read_detector_settings(path):
    from microalgae.imaging.crop_review import DEFAULT, load_detector_settings

    return load_detector_settings(path) if path else dict(DEFAULT)


def load_confirmed_classifier(path, approved_fingerprint, confirm):
    from microalgae.imaging.classifier import ClassifierSelectionCancelled, load_classifier

    try:
        return load_classifier(path, approved_fingerprint, confirm)
    except ClassifierSelectionCancelled:
        return None


class ModelSelection:
    def __init__(self):
        self.settings = QSettings("MicroalgaeID", "MicroalgaeID")
        self.classifier = None

    def add_menu(self, window, *, show_status=None):
        menu = window.menuBar().addMenu("Models")
        detector_status = menu.addAction("")
        detector_status.setEnabled(False)
        menu.addAction("Choose detector settings…", lambda: self.choose_detector(window, show_status=show_status))
        menu.addAction("Use built-in detector", lambda: self.settings.remove("detector"))
        menu.addSeparator()
        classifier_status = menu.addAction("")
        classifier_status.setEnabled(False)
        menu.addAction("Choose classifier folder…", lambda: self.choose_classifier(window, show_status=show_status))
        menu.addAction("Clear classifier", self.clear_classifier)

        def refresh():
            detector = self.settings.value("detector", "", str)
            classifier = self.settings.value("classifier_name", "", str)
            detector_status.setText("Detector · " + (Path(detector).stem if detector else "Built-in hysteresis"))
            classifier_status.setText("Classifier · " + (classifier or "None selected"))

        menu.aboutToShow.connect(refresh)
        refresh()

    def detector_settings(self):
        return read_detector_settings(self.settings.value("detector", "", str))

    def choose_detector(self, window, *, show_status=None):
        from microalgae.imaging.crop_review import load_detector_settings

        path, _ = QFileDialog.getOpenFileName(
            window, "Choose detector settings", self.settings.value("detector", "", str),
            "Detector settings (*.yaml *.yml)",
        )
        if path:
            try:
                load_detector_settings(path)
            except Exception as error:
                QMessageBox.warning(window, "Could not use detector", str(error))
                return
            self.settings.setValue("detector", path)
            report = show_status or window.statusBar().showMessage
            report(f"Detector selected for new batches · {Path(path).stem}", 8000)

    def choose_classifier_path(self, window):
        return QFileDialog.getExistingDirectory(
            window, "Choose the folder containing pipeline.joblib and metadata.json",
            self.settings.value("classifier_path", str(Path.cwd() / "results"), str),
        )

    def choose_classifier(self, window, *, show_status=None):
        path = self.choose_classifier_path(window)
        if path:
            try:
                classifier = self.load(window, path)
            except Exception as error:
                QMessageBox.warning(window, "Could not use classifier", str(error))
                return
            if classifier is not None:
                report = show_status or window.statusBar().showMessage
                report(f"Classifier selected · {classifier.name}", 8000)

    def get_classifier(self, window):
        if self.classifier is not None:
            return self.classifier
        path = self.settings.value("classifier_path", "", str)
        if not path:
            self.choose_classifier(window)
            return self.classifier
        return self.load(window, path)

    def original_classifier(self, window, record):
        if self.classifier is not None and self.classifier.fingerprint == record['fingerprint']:
            return self.classifier
        path = record.get('path')
        if path and Path(path).is_dir():
            try:
                model = self.load(window, path, expected=record['fingerprint'])
            except Exception as error:
                QMessageBox.warning(window, "Original classifier unavailable", str(error))
            else:
                if model is None:
                    return None
                return model
        path = QFileDialog.getExistingDirectory(window, "Locate the original classifier folder")
        if path:
            return self.load(window, path, expected=record['fingerprint'])
        return None

    @staticmethod
    def confirm_classifier(window, model_path):
        return QMessageBox.question(
            window, "Trust this classifier?",
            f"Only open models from people you trust. Loading this file can run code.\n\n{model_path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes

    def load(self, window, path, *, expected=None):
        classifier = load_confirmed_classifier(
            path, self.settings.value("classifier_fingerprint", "", str),
            lambda model_path: self.confirm_classifier(window, model_path),
        )
        if classifier is None:
            return None
        if expected is not None:
            if classifier.fingerprint != expected:
                raise ValueError("This classifier does not match the model saved with the run.")
            return classifier
        self.remember_classifier(path, classifier)
        return classifier

    def remember_classifier(self, path, classifier):
        self.settings.setValue("classifier_path", path)
        self.settings.setValue("classifier_name", classifier.name)
        self.settings.setValue("classifier_fingerprint", classifier.fingerprint)
        self.classifier = classifier

    def clear_classifier(self):
        for key in ("classifier_path", "classifier_name", "classifier_fingerprint"):
            self.settings.remove(key)
        self.classifier = None
