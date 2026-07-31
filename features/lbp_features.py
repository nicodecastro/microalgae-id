import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

# Define a function to extract the local binary patterns features
def extract_features(dataset_dir, class_subfolders, features_dir, save_images=False):
    # Initialize an empty list to store the feature data
    feature_data = []
    labels_data = []

    # Loop over the subfolders and extract features from images in each subfolder
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
            features = extract_lbp_features(image_resize, image_name, subfolder_name, method='uniform', save_images=save_images) # 'default','ror','uniform','nri_uniform','var'
            feature_data.append(features)
            labels_data.append(subfolder_name)
                
    df = pd.DataFrame(feature_data, columns=["feature_{}".format(i) for i in range(len(feature_data[0]))])
    df["Class"] = labels_data

    # Normalize the features using z-score normalization
    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

    # Print the first few rows of the dataframe
    print(df)

    # Export the DataFrame to a CSV file
    export_path = Path(features_dir) / "LBP_features.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
    df.to_csv(export_path, index=False)

def extract_lbp_features(image, image_name, subfolder_name, method='uniform', radius=3, n_points=8, save_images=False):
    lbp = local_binary_pattern(image, n_points, radius, method=method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    if save_images:
        success_path = Path(f"data/success/lbp/{subfolder_name}")
        success_path.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
        cv2.imwrite(str(os.path.join(success_path, f"{Path(image_name).stem}.jpg")), image)

    return hist