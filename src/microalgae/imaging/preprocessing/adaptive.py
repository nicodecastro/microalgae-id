import cv2

def adaptive_threshold(image, block_size=15, c=3):
    # Apply adaptive thresholding based on Chong et al., 2024 paper
    return cv2.adaptiveThreshold(
        image, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        block_size, c
    )
