import cv2
import numpy as np

def resize_with_padding(
    image,
    target_size=(512, 512),
    pad_color=0,
    *,
    return_content_mask=False,
):
    """Resize with padding; optionally return a Boolean image-content mask."""
    if not isinstance(return_content_mask, bool):
        raise ValueError("return_content_mask must be Boolean")
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

    if return_content_mask:
        content_mask = np.zeros((target_h, target_w), dtype=bool)
        content_mask[y:y+new_h, x:x+new_w] = True
        return canvas, content_mask
    return canvas
