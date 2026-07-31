import cv2
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

# Define a function to extract the morphological features
def extract_features(dataset_dir, class_subfolders, features_dir, save_images=False):
    # Initialize an empty list to store the feature data
    feature_data = []

    count = 0
    # Loop over the subfolders and extract features from images in each subfolder
    for subfolder_name in class_subfolders:
        subfolder_path = os.path.join(dataset_dir, subfolder_name)
        subfolder_images = os.listdir(subfolder_path)
        print(f"Class: {subfolder_name} ({len(subfolder_images)} images)")

        # Loop over the images in the subfolder and extract the features
        for image_name in subfolder_images:
            # Load the image
            image_path = os.path.join(subfolder_path, image_name)
            image = cv2.imread(image_path)

            # Extract the features from the image
            features = extract_features_from_image(image, image_name, subfolder_name, save_images)
            # Print image names with features that were failed to be extracted
            if features == None:
              count += 1
              print(f"{count} {subfolder_name} {image_name}")
              feature_data.append(features + [subfolder_name])

            # If features were successfully extracted, add them to the feature data
            if features is not None:
                feature_data.append(features + [subfolder_name])

    # Define the column names for the dataframe
    column_names = [
      "Eccentricity", 
      # "Convexity", 
      # "FeretDiameter", 
      "Area", 
      "Perimeter", 
      # "Solidity", 
      # "Extent", 
      "AspectRatio", 
      "Circularity", 
      "Class"]

    # Create a pandas dataframe from the feature data
    df = pd.DataFrame(feature_data, columns=column_names)

    # ensure numeric
    df['Eccentricity'] = pd.to_numeric(df['Eccentricity'], errors='coerce')

    # Group the dataframe by class
    grouped = df.groupby('Class')

    # Fill NaN values in Eccentricity column with mean value of the same class
    df['Eccentricity'] = grouped['Eccentricity'].transform(lambda x: x.interpolate())

    # Take the mean of the forward and back rows to fill NaN values
    df['Eccentricity'] = (df['Eccentricity'].ffill() + df['Eccentricity'].bfill()) / 2

    # Fill remaining NaN values with median
    df_numeric = df.select_dtypes(include='number')  # select numeric columns only
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.median())

    # Apply Z-score normalization to standardize the features
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    
    # Print the dataframe
    print(df)

    # Export the DataFrame to a CSV file
    export_path = Path(features_dir) / "Geometric_shape_features.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
    df.to_csv(export_path, index=False)
    

def extract_features_from_image(image, image_name, subfolder_name, save_images=False):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.bilateralFilter(gray, 40, 1, 25)
    
    # Apply adaptive thresholding to create a binary image
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Canny Edge detection
    canny = cv2.Canny(blurred, 100, 170)

    # Find the contours in the binary image
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize an empty list to store the features for each contour
    features_list = []

    # Loop over the contours and extract the features
    for cnt in contours:
        # Check if the contour has at least five points
        if len(cnt) < 5:
            continue

        # Calculate the area, perimeter, and Feret diameter of the contour
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        _, (w, h), _ = cv2.minAreaRect(cnt)
        #feret_diameter = max(w, h)

        # Calculate the solidity of the contour
        # hull = cv2.convexHull(cnt)
        # hull_area = cv2.contourArea(hull)
        # if hull_area == 0:
        #     solidity = 0
        # else:
        #     solidity = float(area) / hull_area

        # Calculate the extent of the contour
        # rect = cv2.boundingRect(cnt)
        # rect_area = rect[2] * rect[3]
        # if rect_area == 0:
        #     extent = 0
        # else:
        #     extent = float(area) / rect_area

        # Calculate the aspect ratio of the contour
        aspect_ratio = float(w) / h if w > h else float(h) / w

        # Calculate the circularity of the contour
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Calculate the eccentricity and convexity of the contour
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        if ma == 0:
            eccentricity = 0
        else:
            eccentricity = np.sqrt(1 - (MA / ma) ** 2)
        # convexity = float(area) / cv2.contourArea(hull)

        # Add the features to the list
        features_list.append([
          eccentricity, 
          # convexity, 
          # feret_diameter, 
          area, 
          perimeter, 
          # solidity, 
          # extent, 
          aspect_ratio, 
          circularity
        ])

    if save_images:
        fail_path = Path(f"data/failed/geometric/{subfolder_name}")
        fail_path.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
        success_path = Path(f"data/success/geometric/{subfolder_name}")
        success_path.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed

    # If there are no contours, return None
    if not features_list:
        if save_images:
            cv2.imwrite(str(os.path.join(fail_path, f"{Path(image_name).stem}.jpg")), canny)
        # print(contours)
        return [np.nan] * len(features_list)
    
    if save_images:
        cv2.imwrite(str(os.path.join(success_path, f"{Path(image_name).stem}.jpg")), canny)

    # Otherwise, calculate the average of the features for all the contours
    features_avg = np.mean(features_list, axis=0)

    # Return the average features
    return features_avg.tolist()