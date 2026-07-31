import cv2

def threshold(image, threshold_method):
    if image is None:
        raise ValueError("Input image is None")

    if threshold_method == "adaptive":
      # Apply adaptive thresholding based on Chong et al., 2024 paper
      thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    else:
      thresh = image
    return thresh