import cv2

def bilateral_filter(image, d, sigma_color, sigma_space):
    return cv2.bilateralFilter(
        image, 
        d, 
        sigma_color, 
        sigma_space
    )