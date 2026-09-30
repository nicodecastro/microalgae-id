import cv2

def adaptive_threshold(image):
    # Apply adaptive thresholding based on Chong et al., 2024 paper
    return cv2.adaptiveThreshold(
        image, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        15, 3 # block size, C
    )