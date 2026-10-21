"""
sanchez_matlab.py
---------------
Python implementation of the MATLAB segmentation function by Sanchez et al. (2019)
(https://www.mathworks.com/matlabcentral/fileexchange/70066-segmentation)
 
Pipeline
--------
1. Grayscale conversion (if needed)
2. Binarisation - Otsu threshold
3. Dilation with a disk structuring element
4. Hole filling
5. Erosion with the same structuring element
6. Select the largest connected component
7. Preserve the prepared image grid
8. Contour / edge extraction
 
Dependencies
------------
    pip install numpy scikit-image scipy opencv-python
"""
 
import numpy as np
from skimage import color, filters, morphology, measure
from skimage.morphology import disk, binary_dilation, binary_erosion
from scipy.ndimage import binary_fill_holes
import cv2
from .resize import resize_with_padding
 
 
def segment_image(
    image: np.ndarray, dilation_size: int = 4, save_images: bool = False, *,
    resize=(256, 256),
):
    """
    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale H×W  or  colour H×W×3 / H×W×4).
    dilation_size : int
        Radius (in pixels) of the disk structuring element used for
        both dilation and erosion.  Typical values: 2, 4, 15.
    resize : tuple or None
        Preserve the legacy padded 256 x 256 default. Pass None when the input
        has already been prepared on its final grid.
 
    Returns
    -------
    mask : np.ndarray (bool, 2-D)
        Binary segmentation mask of the largest connected component on the
        resized grid, or the input grid when resize=None.
    image_cut : np.ndarray
        Grayscale intensities with background masked out, on the same grid.
    image_contour : np.ndarray (bool, 2-D)
        Edge / contour image derived from the mask (Sobel edges).
 
    Notes
    -----
    The MATLAB code pads the binary image with zeros (2×dilation_size on
    each side) before morphological operations so that objects touching
    the border are handled correctly.  This port replicates that behaviour
    with numpy.pad.
    """
    # ------------------------------------------------------------------
    # 0. Resize & pad
    # ------------------------------------------------------------------
    if resize is not None:
        image = resize_with_padding(image, resize)
 
    # ------------------------------------------------------------------
    # 1. Convert to grayscale if necessary
    # ------------------------------------------------------------------
    if image.ndim == 3:
        if image.shape[2] == 4:                   # RGBA → drop alpha
            image = image[..., :3]
        gray = color.rgb2gray(image)              # float64, range [0, 1]
    else:
        gray = image.astype(np.float64)
        if gray.max() > 1.0:
            gray = gray / 255.0
 
    # ------------------------------------------------------------------
    # 2. Binarise with Otsu threshold and invert
    #    MATLAB: imageBW = imbinarize(image);  imageBW = not(imageBW);
    # ------------------------------------------------------------------
    thresh = filters.threshold_otsu(gray)
    binary = gray < thresh          # invert: dark objects become True
 
    # ------------------------------------------------------------------
    # 3. Pad to avoid border artefacts during morphology
    #    MATLAB: padarray(imageBW, [2*d 2*d], 0, 'both')
    # ------------------------------------------------------------------
    pad = 2 * dilation_size
    binary = np.pad(binary, pad_width=pad, mode='constant', constant_values=0)
 
    # ------------------------------------------------------------------
    # 4. Dilate with a disk structuring element
    #    MATLAB: strel('disk', dilationSize, 8)  →  skimage disk(r)
    # ------------------------------------------------------------------
    se = disk(dilation_size)
    binary = binary_dilation(binary, footprint=se)
 
    # ------------------------------------------------------------------
    # 5. Fill holes
    #    MATLAB: imfill(imageBW, 'holes')
    # ------------------------------------------------------------------
    binary = binary_fill_holes(binary)
 
    # ------------------------------------------------------------------
    # 6. Erode
    #    MATLAB: imerode(imageBW, se)
    # ------------------------------------------------------------------
    binary = binary_erosion(binary, footprint=se)
 
    # ------------------------------------------------------------------
    # 7. Select the largest connected component
    #    MATLAB: bwconncomp → regionprops('Area') → max area index
    # ------------------------------------------------------------------
    labeled = measure.label(binary)
    if labeled.max() == 0:
        # No foreground found – return empty results
        empty = np.zeros_like(gray, dtype=bool)
        return empty, gray, empty
 
    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
 
    # Build mask containing only the largest region
    mask_full = labeled == largest.label
 
    # ------------------------------------------------------------------
    # 8. DO NOT CROP — keep fixed grid for Log-Gabor compatibility
    #    Instead of MATLAB imcrop, we preserve full image size
    #    and apply mask only.
    # ------------------------------------------------------------------

    # Remove morphology padding
    mask = mask_full[pad:-pad, pad:-pad]

    # Apply mask to original grayscale image (fixed 256x256 canvas)
    image_cut = gray * mask.astype(np.float64)
 
    # # ------------------------------------------------------------------
    # # 8. Crop to bounding box
    # #    MATLAB: statsBB(index).BoundingBox  →  regionprops BoundingBox
    # #    skimage BoundingBox: (min_row, min_col, max_row, max_col)
    # #    We also need to crop the *original* grayscale (un-padded) image,
    # #    so we first remove the padding from both the mask and the coords.
    # # ------------------------------------------------------------------
    # min_row, min_col, max_row, max_col = largest.bbox
 # 
    # # Crop mask and remove the padding we added earlier
    # mask = mask_full[min_row:max_row, min_col:max_col]
 # 
    # # Map bounding-box coordinates back to the un-padded gray image
    # orig_min_row = max(min_row - pad, 0)
    # orig_min_col = max(min_col - pad, 0)
    # orig_max_row = max_row - pad
    # orig_max_col = max_col - pad
 # 
    # image_cut = gray[orig_min_row:orig_max_row, orig_min_col:orig_max_col]
 
    # ------------------------------------------------------------------
    # 9. Contour / edge extraction
    #    MATLAB: edge(mask)  →  Sobel edges on the uint8 mask
    # ------------------------------------------------------------------
    image_contour = filters.sobel(mask.astype(float)) > 0
 
    if save_images:
        mask = (mask * 255).astype(np.uint8)
        image_cut = (image_cut * 255).astype(np.uint8)
        image_contour = (image_contour * 255).astype(np.uint8)
 
    return mask, image_cut, image_contour
 
 
# ---------------------------------------------------------------------------
# Quick demo – run this file directly to test with a sample image
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
 
    # Load an image (pass a path as CLI arg, or fall back to a test image)
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Use skimage's built-in "coins" image for a quick self-test
        from skimage.data import coins
        img = coins()
 
    mask, img_cut, contour = segment_image(img, dilation_size=4)
 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mask, cmap="gray");       axes[0].set_title("Mask")
    axes[1].imshow(img_cut, cmap="gray");    axes[1].set_title("Cropped image")
    axes[2].imshow(contour, cmap="gray");    axes[2].set_title("Contour")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
 
