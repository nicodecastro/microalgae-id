import cv2

def canny(image, low_threshold=150, high_threshold=175):
    if image is None:
        raise ValueError("Input image is None")

    # Apply adaptive thresholding based on Chong et al., 2024 paper
    segmented = cv2.Canny(image, low_threshold, high_threshold)

    return segmented
