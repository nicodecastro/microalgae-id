import cv2

def segment(image, segment_method):
    if image is None:
        raise ValueError("Input image is None")

    if segment_method == "canny":
      # Apply adaptive thresholding based on Chong et al., 2024 paper
      segmented = cv2.Canny(image, 150, 175)
    elif segment_method == "sobel":
        # Apply Sobel operator to the image
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        image_sobel_x = cv2.convertScaleAbs(sobelx)
        image_sobel_y = cv2.convertScaleAbs(sobely)
        # image_sobel_mag = cv2.magnitude(sobelx, sobely)
        image_sobel_bit = cv2.bitwise_or(sobelx, sobely)
    else:
      segmented = image
    return segmented