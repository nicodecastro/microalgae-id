import cv2
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path

# Define function to extract flattened Hu moments from an image
def extract_features(dataset_dir, class_subfolders, features_dir, save_images=False):
    # Initialize empty list to store extracted features
    feature_data = []

    count = 0
    # Loop over subdirectories in the image directory
    for subfolder_name in class_subfolders:
        subfolder_path = os.path.join(dataset_dir, subfolder_name)
        subfolder_images = os.listdir(subfolder_path)
        print(f"Class: {subfolder_name} ({len(subfolder_images)} images)")

        # Loop over the images in the subfolder and extract the features
        for image_name in subfolder_images:
            # Load the image
            image_path = os.path.join(subfolder_path, image_name)
            
            # Extract mean Hu moments from image
            hu_moments_mean = extract_hu_moments(image_path, image_name, subfolder_name, save_images)

            # Append class label and mean Hu moments to features list
            feature_data.append(hu_moments_mean + [subfolder_name])

    # Create pandas DataFrame from features list
    columns = [f'Hu_{i}' for i in range(1, 8)] + ['Class']
    df = pd.DataFrame(feature_data, columns=columns)

    # Choose which scaler to use: 'minmax' for MinMaxScaler, 'standard' for StandardScaler
    scaler_type = 'standard'

    # Normalize the Hu moments using chosen scaler
    hu_moments = df.iloc[:, :-1].values
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(hu_moments)
    hu_moments_normalized = scaler.transform(hu_moments)

    # Add the normalized Hu moments to the DataFrame
    for i in range(7):
        df[f'Hu_{i+1}'] = hu_moments_normalized[:, i]

    # Display the DataFrame
    print(df)

    # Export the DataFrame to a CSV file
    if scaler_type == 'minmax':
        export_path = Path(features_dir) / "Hu_moments_minmax_features.csv"
    else:
        export_path = Path(features_dir) / "Hu_moments_nonminmax_features.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
    df.to_csv(export_path, index=False)

def extract_hu_moments(image_path, image_name, subfolder_name, save_images=False):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply bilateral blur to smooth the image while preserving edges
    blurred = cv2.bilateralFilter(img, 40, 1, 25)

    # Apply adaptive thresholding to create a binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract flattened Hu moments for each contour
    hu_moments_list = []
    for contour in contours:
        # Calculate moments for the contour
        moments = cv2.moments(contour)
        # Calculate Hu moments for the moments
        hu_moments = cv2.HuMoments(moments)
        # Flatten the Hu moments array
        hu_moments_flat = hu_moments.flatten()
        # Append the flattened Hu moments to the list
        hu_moments_list.append(hu_moments_flat)

    if save_images:
        fail_path = Path(f"data/failed/hu_moments/{subfolder_name}")
        fail_path.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
        success_path = Path(f"data/success/hu_moments/{subfolder_name}")
        success_path.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed

    # If no contours were found, return a list of zeros
    if len(hu_moments_list) == 0:
        if save_images:
            cv2.imwrite(str(os.path.join(fail_path, f"{Path(image_name).stem}.jpg")), thresh)
        return [0]*7

    if save_images:
        cv2.imwrite(str(os.path.join(success_path, f"{Path(image_name).stem}.jpg")), thresh)

    # Calculate the mean Hu moments for all contours in the image
    mean_hu_moments = np.mean(hu_moments_list, axis=0)

    # Return the mean Hu moments for the image
    return mean_hu_moments.tolist()