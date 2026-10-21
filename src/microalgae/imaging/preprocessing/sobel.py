import cv2

def sobel(image, ksize=3):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    magnitude = cv2.magnitude(sobelx, sobely)
    image_sobel = cv2.convertScaleAbs(magnitude)

    return image_sobel
