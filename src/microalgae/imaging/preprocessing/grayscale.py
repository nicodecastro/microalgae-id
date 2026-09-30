import cv2

def to_grayscale(image):
    if image is None:
        raise ValueError("Input image is None")

    # If already grayscale, just return
    if len(image.shape) == 2:
        return image

    # Convert BGR → Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray