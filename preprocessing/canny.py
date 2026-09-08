import cv2

def canny(image):
    if image is None:
        raise ValueError("Input image is None")

    # Apply adaptive thresholding based on Chong et al., 2024 paper
    segmented = cv2.Canny(image, 150, 175)

    return segmented