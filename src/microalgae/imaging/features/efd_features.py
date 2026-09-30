import os
import cv2
import numpy as np
import pandas as pd
from pyefd import elliptic_fourier_descriptors
from pathlib import Path
from ..preprocessing import segment_image

# Define a function to extract the efd features
def extract_features(dataset_dir, class_subfolders, features_dir, save_images=False):
    # Initialize an empty list to store the feature data
    feature_data = []
    labels_data = []

    # EFD Order
    num_harmonics = 30
    
    # Define the EFD properties to extract
    properties = []
    for n in range(1, num_harmonics + 1):
        properties.append(f"efd_amp{n}")
    
    # Loop through the subfolders and extract features from each image
    for subfolder_name in class_subfolders:
        subfolder_path = os.path.join(dataset_dir, subfolder_name)
        subfolder_images = os.listdir(subfolder_path)
        print(f"Class: {subfolder_name} ({len(subfolder_images)} images)")

        # Loop over the images in the subfolder and extract the features
        for image_name in subfolder_images:
            # Load the image
            image_path = os.path.join(subfolder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Preprocess images based on Sanchez et al. (2024)
            mask, image_cut, image_contour = segment_image(image, dilation_size=4)
            
            # Extract the features from the image
            features = extract_efd_features(mask, image_name, subfolder_name, save_images)
            feature_data.append(features)
            labels_data.append(subfolder_name)
            
    # Convert the features and labels to a pandas dataframe
    df = pd.DataFrame(feature_data, columns=properties)
    df['Class'] = labels_data

    # Print the first few rows of the dataframe
    print(df)

    # Export the DataFrame to a CSV file
    export_path = Path(features_dir) / "EFD_features.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
    df.to_csv(export_path, index=False)

def extract_efd_features(image, image_name, subfolder_name, save_images=False):
    # Find the contours of a binary image using OpenCV.
    mask_uint8 = (image * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_cnt = max(contours, key=cv2.contourArea)

    coeff = elliptic_fourier_descriptors(np.squeeze(largest_cnt), order=30)
    feature = []
    feature.extend(0.5 * np.sqrt(np.sum(coeff**2, axis=1)))
    
    if save_images:
        success_path = Path(f"data/success/efd/{subfolder_name}")
        success_path.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
        cv2.imwrite(str(os.path.join(success_path, f"{Path(image_name).stem}.jpg")), image)
    
    return feature
