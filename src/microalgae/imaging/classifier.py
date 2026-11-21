"""Load a local classifier with the feature recipe saved at training time."""

from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import yaml

from .features import (
    efd_features as efd,
    gabor_features as gabor,
    geometric_features as geom,
    glcm_features as glcm,
    hu_moments_features as hu,
    lbp_features as lbp,
    phase_congruency_features as pc,
)
from .preprocessing.configured import prepare_feature_input, validate_preprocessing


FEATURES = {
    "efd": (efd.FEATURE_NAMES, lambda p: efd.extract_efd_features(p.mask, "crop", "")),
    "gabor": (gabor.FEATURE_NAMES, lambda p, bank: gabor.extract_gabor_statistics(p.image, p.mask, lg=bank)),
    "geom": (geom.FEATURE_NAMES, lambda p: geom.extract_geometric_features(p.image)),
    "glcm": (glcm.PROPERTIES, lambda p: glcm.extract_glcm_statistics(p.image)),
    "hu": (hu.FEATURE_NAMES, lambda p: hu.extract_hu_moments_from_binary(p.image)),
    "lbp": (lbp.FEATURE_NAMES, lambda p: lbp.extract_lbp_histogram(p.image)),
    "pc": (pc.pc_feature_names, lambda p: pc.phase_congruency_features(p.image)),
}


class FeatureClassifier:
    """Use a fitted scikit-learn pipeline with the project's raw descriptors."""

    def __init__(self, directory, metadata, metadata_bytes, approved, confirm):
        recipe = directory / "resolved_settings.yaml"
        if not recipe.is_file() and directory.name == "final" and directory.parent.parent.name == "evaluation":
            recipe = directory.parent.parent.parent / recipe.name
        recipe_bytes = recipe.read_bytes()
        settings = yaml.safe_load(recipe_bytes)
        if settings["standardize"] is not False:
            raise ValueError("Choose a classifier trained on raw features with its fitted scaler.")
        self.preprocessing = {
            family: validate_preprocessing(options, family)
            for family, options in settings["resolved_preprocessing"].items()
        }
        self.feature_columns = metadata["feature_columns"]
        expected = {
            f"{family}__{column}"
            for family in self.preprocessing
            for column in FEATURES[family][0]
        }
        if set(self.feature_columns) != expected or len(self.feature_columns) != len(expected):
            raise ValueError("Model feature columns and saved preprocessing do not match.")
        self.classes = metadata["estimator_classes"]
        if not self.classes or set(self.classes) != set(metadata["class_labels"]):
            raise ValueError("Model metadata has inconsistent species labels.")
        saved_version = metadata["library_versions"]["scikit-learn"]
        installed_version = version("scikit-learn")
        if installed_version != saved_version:
            raise ValueError(f"This classifier uses scikit-learn {saved_version}. "
                             f"The app is running {installed_version}.")
        model_path = directory / "pipeline.joblib"
        model_bytes = model_path.read_bytes()
        self.fingerprint = sha256(b"".join(
            sha256(part).digest() for part in (metadata_bytes, recipe_bytes, model_bytes)
        )).hexdigest()
        self.name = (
            f"{directory.parent.parent.parent.name} / {directory.parent.name.upper()}"
            if directory.name == "final" and directory.parent.parent.name == "evaluation"
            else directory.name
        )
        if self.fingerprint != approved and not confirm(model_path):
            raise ClassifierSelectionCancelled()
        self.pipeline = joblib.load(BytesIO(model_bytes))
        if (not callable(getattr(self.pipeline, "predict", None))
                or getattr(self.pipeline, "n_features_in_", None) != len(self.feature_columns)
                or list(getattr(self.pipeline, "classes_", ())) != self.classes):
            raise ValueError("The fitted classifier does not match its saved feature count or species.")
        self._gabor_bank = None
        self._gabor_bank_lock = Lock()

    def _get_gabor_bank(self):
        with self._gabor_bank_lock:
            if self._gabor_bank is None:
                self._gabor_bank = gabor._LogGaborBank()
            return self._gabor_bank

    def features(self, image):
        values = {}
        for family, settings in self.preprocessing.items():
            columns, calculate = FEATURES[family]
            prepared = prepare_feature_input(image, settings, family)
            if family == "gabor":
                result = calculate(prepared, self._get_gabor_bank())
            else:
                result = calculate(prepared)
            features = np.asarray(result, dtype=float)
            if features.shape != (len(columns),) or not np.isfinite(features).all():
                raise ValueError(f"Could not extract valid {family} features from this crop.")
            values.update((f"{family}__{column}", value) for column, value in zip(columns, features))
        return np.array([[values[column] for column in self.feature_columns]])

    def predict(self, image):
        label, = self.pipeline.predict(self.features(image))
        if label not in self.classes:
            raise ValueError("Classifier returned an unexpected species.")
        return str(label)


class ClassifierSelectionCancelled(Exception):
    """The user declined to open a model file."""


# A different input/model format can provide the same name, fingerprint, and predict(image) interface.
CLASSIFIERS = {"sklearn_features": FeatureClassifier}


def load_classifier(directory, approved, confirm):
    directory = Path(directory).resolve()
    metadata_bytes = (directory / "metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    name = metadata.get("classifier_format", "sklearn_features")
    if name not in CLASSIFIERS:
        raise ValueError(f"Classifier format {name!r} is unavailable in this app.")
    return CLASSIFIERS[name](directory, metadata, metadata_bytes, approved, confirm)
