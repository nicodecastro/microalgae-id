from pathlib import Path
from preprocessing import resize_with_padding
from utils.io import load_image_paths, save_image
from utils.config import load_config
import cv2

# -----------------------------
# Load config
# -----------------------------
config = load_config()
RAW_DIR = config["dataset"]["raw_dir"]
PROCESSED_DIR = config["dataset"]["processed_dir"]
TARGET_SIZE = config["preprocessing"]["resize"]["target_size"]
CLASSES = config["preprocessing"]["classes"]

# Ensure class folders exist in raw and processed
for img_class in CLASSES:
  folder_path = PROCESSED_DIR+img_class
  if Path(folder_path).is_dir():
    print(f"Found '{folder_path}'.")
  else:
    print(f"Error: The folder '{folder_path}' does not exist or is not a directory.")
    exit(1)

# Load all raw images
image_paths = load_image_paths(RAW_DIR, extension=".png")

for path in image_paths:
    # Load image
    img = cv2.imread(str(path))

    # Preprocessing pipeline
    img = resize_with_padding(img, target_size=TARGET_SIZE)
    # img = to_grayscale(img)
    # img = bilateral_filter(img)

    # Save to processed folder
    save_path = Path(PROCESSED_DIR) / path.name
    save_image(save_path, img)