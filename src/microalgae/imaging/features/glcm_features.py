import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

# Define a function to extract the glcm features
def extract_features(dataset_dir, class_subfolders, features_dir, save_images=False):
    # Initialize an empty list to store the feature data
    feature_data = []
    labels_data = []

    # Define the GLCM properties to extract
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
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
            image_resize = cv2.resize(image, (512, 512))
            
            # Extract the features from the image
            features = extract_glcm_features(image_resize, image_name, subfolder_name, properties, save_images)
            feature_data.append(features)
            labels_data.append(subfolder_name)
            
    # Convert the features and labels to a pandas dataframe
    df = pd.DataFrame(feature_data, columns=properties)
    df['Class'] = labels_data

    # Normalize the features using z-score normalization
    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

    # Print the first few rows of the dataframe
    print(df)

    # Export the DataFrame to a CSV file
    export_path = Path(features_dir) / "GLCM_features.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
    df.to_csv(export_path, index=False)

def extract_glcm_features(image, image_name, subfolder_name, properties, save_images=False):
    # Define the distances and angles to consider
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # Angle at 0, 45, 90, 135
    
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    feature = [graycoprops(glcm, prop).ravel()[0] for prop in properties]
    
    if save_images:
        success_path = Path(f"data/success/glcm/{subfolder_name}")
        success_path.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
        cv2.imwrite(str(os.path.join(success_path, f"{Path(image_name).stem}.jpg")), image)
    
    return feature