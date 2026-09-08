import cv2

def sobel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(sobelx, sobely)
    image_sobel = cv2.convertScaleAbs(magnitude)

    return image_sobel