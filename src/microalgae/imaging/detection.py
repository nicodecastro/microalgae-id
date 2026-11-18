"""Class-agnostic source-pixel region proposals."""
from math import ceil, floor
import cv2
import numpy as np
from .preprocessing.geometry_foreground import geometry_foreground_mask


def detect_regions(image, *, low_ratio, background_kernel_size, min_component_area,
                   max_side=512, margin_ratio=0.1):
    if image.dtype != np.uint8 or image.ndim not in (2, 3) or not image.size:
        raise ValueError('Expected a nonempty uint8 image')
    if max_side < 1 or not 0 <= margin_ratio <= 1:
        raise ValueError('Invalid working size or margin')
    h, w = image.shape[:2]
    scale = min(1, max_side / max(h, w))
    rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA)
    mask = geometry_foreground_mask(gray, np.ones(gray.shape, bool),
        method='otsu_background_hysteresis', hysteresis_low_ratio=low_ratio,
        background_kernel_size=background_kernel_size, min_component_area=min_component_area)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=8)
    regions = []
    for x, y, width, height, area in stats[1:]:
        box = [floor(x*w/rw), floor(y*h/rh), min(w, ceil((x+width)*w/rw)), min(h, ceil((y+height)*h/rh))]
        l, t, r, b = box
        margin = max(2, ceil(max(r-l, b-t)*margin_ratio))
        regions.append(dict(box=box, crop_box=[max(0,l-margin), max(0,t-margin), min(w,r+margin), min(h,b+margin)],
                            component_area=int(area), truncated=l==0 or t==0 or r==w or b==h))
    regions.sort(key=lambda o: (o['box'][1], o['box'][0], o['box'][3], o['box'][2]))
    return dict(regions=regions, transform=dict(source_size=[w,h], working_size=[rw,rh], scale_x=rw/w, scale_y=rh/h))
