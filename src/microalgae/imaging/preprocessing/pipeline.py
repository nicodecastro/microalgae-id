from .resize import resize_with_padding
from .grayscale import to_grayscale
from .bilateral import bilateral_filter
from .adaptive import adaptive_threshold
from .canny import canny
from .sobel import sobel

import numpy as np


def preprocess_image(
    image,
    resize=(256, 256),
    grayscale=True,
    denoise=None,
    bilateral_params=None,
    threshold=None,
    edge=None,
):
    """
    Generic preprocessing pipeline.

    Parameters
    ----------
    image : np.ndarray

    resize : tuple or None
        Target image size. Pass ``None`` to retain the original size.

    denoise : str
        none | bilateral

    bilateral_params : tuple or None
        ``(diameter, sigma_color, sigma_space)`` for bilateral filtering.

    threshold : str
        none | adaptive

    edge : str
        none | sobel | canny

    Returns
    -------
    np.ndarray
    """

    if resize is not None:
        image = resize_with_padding(image, resize)
    
    if grayscale:
        image = to_grayscale(image)

    if denoise == "bilateral":
        params = bilateral_params or (9, 75, 75)
        image = bilateral_filter(image, *params)

    if threshold == "adaptive":
        image = adaptive_threshold(image)
        
    if edge == "sobel":
        image = sobel(image) 
    elif edge == "canny":
        image = canny(image)

    return image
