"""Image-coordinate results accepted by the review interface."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal


@dataclass(frozen=True, slots=True)
class Detection:
    """A pixel-coordinate box; model scores display only when explicitly validated."""

    identifier: str
    x: float
    y: float
    width: float
    height: float
    species: str
    probability: float | None = None
    probability_validated: bool = False

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not self.identifier or not self.species:
            raise ValueError("A detection needs an identifier and species label")
        if not all(isfinite(value) for value in values):
            raise ValueError("Detection coordinates must be finite")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Detection box must have a nonnegative origin and positive size")
        if self.probability is not None and (
            not isfinite(self.probability) or not 0 <= self.probability <= 1
        ):
            raise ValueError("Probability must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ImageAnalysis:
    """Results for one image, with demo output distinguishable from model output."""

    image_id: str
    detections: tuple[Detection, ...]
    source: Literal["demo", "model"]

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("An analysis needs an image identifier")
        identifiers = [detection.identifier for detection in self.detections]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Detection identifiers must be unique within an image")
