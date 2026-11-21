"""Qt worker and display conversion for saved prediction runs."""

from copy import deepcopy
from threading import Event

from PySide6.QtCore import QObject, Signal

from microalgae.domain.results import Detection, ImageAnalysis
from microalgae.ui.models import load_confirmed_classifier, read_detector_settings


class PredictionTask(QObject):
    trust_requested = Signal(object)
    prepared = Signal(object)
    progress = Signal(int, object)
    finished = Signal(object)

    def __init__(self, directory, images, *, detector_path, classifier, model_path, approved_fingerprint):
        super().__init__()
        self.directory = directory
        self.images = images
        self.detector_path = detector_path
        self.classifier = classifier
        self.model_path = model_path
        self.approved_fingerprint = approved_fingerprint
        self.stop = Event()
        self._trust_answered = Event()
        self._trusted = False

    def cancel(self):
        self.stop.set()
        self._trust_answered.set()

    def answer_trust(self, trusted):
        self._trusted = trusted
        self._trust_answered.set()

    def _request_trust(self, path):
        if self.stop.is_set():
            return False
        self.trust_requested.emit(path)
        self._trust_answered.wait()
        return self._trusted and not self.stop.is_set()

    def run(self):
        error = None
        try:
            if self.stop.is_set():
                return
            from microalgae.imaging.prediction import run_predictions

            settings = read_detector_settings(self.detector_path)
            if self.stop.is_set():
                return
            classifier = self.classifier
            if classifier is None:
                classifier = load_confirmed_classifier(self.model_path, self.approved_fingerprint, self._request_trust)
            if classifier is None:
                self.cancel()
                return
            if self.stop.is_set():
                return
            self.prepared.emit(classifier)
            run_predictions(
                self.directory, self.images, settings, classifier, self.model_path, self.stop,
                lambda index, row: self.progress.emit(index, deepcopy(row)),
            )
        except Exception as failure:
            error = str(failure)
        finally:
            self.finished.emit(error)


def image_analysis(row):
    status = {
        'failed': f"Image failed: {row['error']}",
        'pending': 'Not processed',
        'stopped': 'Stopped before all crops were predicted',
    }.get(row['status'], '')
    detections = []
    for crop in row['crops']:
        if not crop['active']:
            continue
        result = crop.get('classification', {})
        species = result.get('species')
        detail = 'Unreviewed' if species else 'Needs prediction'
        if 'error' in result:
            detail = f"Prediction failed: {result['error']}"
        left, top, right, bottom = crop['box']
        detections.append(Detection(crop['crop_id'], left, top, right-left, bottom-top,
                                    species, status=detail))
    if not detections and not status:
        status = 'No crops detected' if row['status'] == 'zero proposals' else 'No active crops'
    return ImageAnalysis(
        row['image_id'], tuple(detections), 'model', status,
        prediction_seconds=row.get('prediction_seconds'),
    )
