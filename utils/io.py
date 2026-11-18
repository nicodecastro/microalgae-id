import cv2
import csv
from pathlib import Path
import os.path
import numpy as np

def load_image(path, grayscale=False):
  img = cv2.imread(str(path))
  if img is None:
    raise ValueError(f"Failed to load image: {path}")

  if grayscale:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  return img
  
def save_image(path, filename, image, printStatus=False):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    full_path = path / f"{filename}.jpg"

    if printStatus:
      print(f"Saving to {full_path}", end=": ")
      success = cv2.imwrite(str(full_path), image)
      print(f"{success}")

def load_image_paths(root_dir, labels, extensions):
  '''
  Load image paths into a dictionary of label: [image_paths]
  '''
  labels_path_dict = dict()
  print("Loading image paths...")
  for label in labels:
    class_dir_list = []
    class_dir = os.path.join(root_dir, label)
    for extension in extensions:
      class_dir_list.extend(list(Path(class_dir).rglob(f"*{extension}")))
    labels_path_dict[label] = class_dir_list
  print(f"Finished loading image paths.\n")
  print(f"Image Count per Label.")
  for key, lst in labels_path_dict.items():
    print(f"{key}: {len(lst)} images")
  print()
  return labels_path_dict
  
def load_labels(csv_path):
  labels = {}
  with open(csv_path, newline="") as f:
    reader = csv.reader(f)
    for image_id, label in reader:
      labels[image_id] = label
  return labels

def save_features(path, features, labels):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, X=features, y=labels)

def load_features(path):
  data = np.load(path)
  return data["X"], data["y"]