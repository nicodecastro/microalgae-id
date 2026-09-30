import cv2
import numpy as np

def resize_with_padding(
    image,
    target_size=(512, 512),
    pad_color=0
):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if len(image.shape) == 3:  # color
      canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    else:  # grayscale
      canvas = np.zeros((target_h, target_w), dtype=image.dtype)
    canvas[:] = pad_color

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized

    return canvas