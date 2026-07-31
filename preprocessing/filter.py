import cv2
import numpy as np


def filter(
    image,
    filter_method,
    params,
):
    if image is None:
        raise ValueError("Input image is None")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a cv2/numpy image")

    filtered = image

    if filter_method == "bilateral":
      filtered = cv2.bilateralFilter(
          image,
          d=params[0],
          sigmaColor=params[1],
          sigmaSpace=params[2]
      )

    return filtered