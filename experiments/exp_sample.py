import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from features import geometric_features, hu_moments_features, lbp_features, glcm_features, efd_features, gabor_features # phase_congruency_features
from utils.config import load_config  # YAML loader
from classifiers.svm_model import run_svm
from classifiers.knn_model import run_knn

from preprocessing import preprocess_image

def run_experiment(config_path="config.yaml"):
    # -----------------------------
    # Load config
    # -----------------------------
    config = load_config(config_path)

    dataset_dir = config["dataset"]["raw_dir"]
    features_dir = config["dataset"]["features_dir"]
    combined_features_dir = config["dataset"]["combined_features_dir"]
    class_subfolders = config["dataset"]["classes"]
    feature_list = config["features"]["feature_list"]
    feature_data = dict.fromkeys(feature_list)

    # -----------------------------
    # Test Preprocessing Pipeline
    # -----------------------------
    for subfolder_name in class_subfolders:
        subfolder_path = os.path.join(dataset_dir, subfolder_name)
        for image_name in os.listdir(subfolder_path):

            image_path = os.path.join(subfolder_path, image_name)

            image = cv2.imread(image_path)

            resized_orig_image = preprocess_image(
                image,
                resize=(256, 256),
                grayscale=True,
            )

            processed = preprocess_image(
                image,
                resize=(256,256),
                grayscale=True,
                denoise="bilateral",
                threshold="adaptive",
                edge="sobel"
            )
            
            success_path = Path(
                f"data/success/preprocessing/{subfolder_name}"
            )

            success_path.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(
                str(success_path / image_name),
                cv2.hconcat([resized_orig_image, processed])
            )
              
    # -----------------------------
    # Load images and extract features
    # -----------------------------
    # Extract features from the image
    # geom
    #print("Extracting geometric features...")
    #geometric_features.extract_features(dataset_dir, class_subfolders, features_dir, save_images=False)
    # hu
    # print("Extracting hu moments features...")
    # hu_moments_features.extract_features(dataset_dir, class_subfolders, features_dir, save_images=False)
    # lbp
    # print("Extracting LBP features...")
    # lbp_features.extract_features(dataset_dir, class_subfolders, features_dir, save_images=False)
    # glcm
    # print("Extracting GLCM features...")
    # glcm_features.extract_features(dataset_dir, class_subfolders, features_dir, save_images=False)
    # efd
    # print("Extracting EFD features...")
    # efd_features.extract_features(dataset_dir, class_subfolders, features_dir, save_images=False)
    # gabor
    # print("Extracting gabor features...")
    # gabor_features.extract_features(dataset_dir, class_subfolders, features_dir, save_images=False)
    # pc
    
    # -----------------------------
    # Combine features
    # -----------------------------
    # print("Combining features...")
    # check if combined features dir exists
    # output_file = "Combined_features.csv"
    # combined_features_path = Path(combined_features_dir).joinpath(output_file)
    # combined_features_path.parent.mkdir(parents=True, exist_ok=True)
    
    # features_path = Path(features_dir)
    # dfs = []
    # class_col = None
    # for file in features_path.iterdir():  # iterate over all feature csv files
        # if file.is_file():
            # df = pd.read_csv(file)

            # save class column once
            # if class_col is None:
                # class_col = df.iloc[:, -1]

            # store feature columns
            # dfs.append(df.iloc[:, :-1])

    # combine all features horizontally
    # combined = pd.concat(dfs, axis=1)

    # add class column back
    # combined["Class"] = class_col

    #combined.to_csv(combined_features_path, index=False)
    #print(f"Combined CSV saved to {combined_features_path}")

    # -----------------------------
    # Run classifier
    # -----------------------------
    #run_svm(combined)
    #run_knn(combined)

    # # -----------------------------
    # # Cross-validation
    # # -----------------------------
    # kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=experiment_cfg.get("random_seed", 42))
    # scores = []

    # for train_idx, test_idx in kf.split(X_scaled, y):
    #     X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    #     y_train, y_test = y[train_idx], y[test_idx]

    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)

    #     metrics = classification_metrics(y_test, y_pred)
    #     scores.append(metrics["accuracy"])

    # # -----------------------------
    # # Summarize results
    # # -----------------------------
    # summary = summarize_cv(scores)
    # print(f"Pipeline: {experiment_cfg['pipeline_name']}")
    # print(f"Accuracy (5-fold CV): {summary['mean']*100:.2f}% ± {summary['std']*100:.2f}%")

    # return summary


if __name__ == "__main__":
    run_experiment("config.yaml")