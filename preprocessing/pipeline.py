from .resize import resize_with_padding
from .grayscale import to_grayscale
from .bilateral import bilateral_filter
from .adaptive import adaptive_threshold
from .canny import canny
from .sobel import sobel

import cv2


def preprocess_image(
    image,
    resize=(256, 256),
    grayscale=True,
    denoise=None,
    threshold=None,
    edge=None,
):
    """
    Generic preprocessing pipeline.

    Parameters
    ----------
    image : np.ndarray

    resize : tuple
        Target image size.

    denoise : str
        none | bilateral

    threshold : str
        none | adaptive

    edge : str
        none | sobel | canny

    Returns
    -------
    np.ndarray
    """

    image = resize_with_padding(image, resize)
    
    if grayscale:
        image = to_grayscale(image)

    if denoise == "bilateral":
        image = bilateral_filter(image)

    if threshold == "adaptive":
        image = adaptive_threshold(image)
        
    if edge == "sobel":
        image = sobel(image) 
    elif edge == "canny":
        image = canny(image)

    return image