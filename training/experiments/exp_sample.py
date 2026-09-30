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

from microalgae.imaging.features import geometric_features, hu_moments_features, lbp_features, glcm_features, efd_features, gabor_features, phase_congruency_features as pc
from training.config import load_config
from training.classifiers.svm_model import run_svm
from training.classifiers.knn_model import run_knn

from microalgae.imaging.preprocessing import preprocess_image

def _preprocess_from_config(image, config):
    dataset_cfg = config["dataset"]
    preprocessing_cfg = config["preprocessing"]

    resize_cfg = preprocessing_cfg["resize"]
    resize = tuple(resize_cfg["target_size"]) if resize_cfg["enabled"] else None

    filter_cfg = preprocessing_cfg["filter"]
    denoise = filter_cfg["method"] if filter_cfg["enabled"] else None
    bilateral_params = tuple(filter_cfg["bilateral"]["params"])

    threshold_cfg = preprocessing_cfg["threshold"]
    threshold = threshold_cfg["method"] if threshold_cfg["enabled"] else None

    edge_cfg = preprocessing_cfg["segment"]
    edge = edge_cfg["method"] if edge_cfg["enabled"] else None

    return preprocess_image(
        image,
        resize=resize,
        grayscale=dataset_cfg["grayscale"],
        denoise=denoise,
        bilateral_params=bilateral_params,
        threshold=threshold,
        edge=edge,
    )


def run_experiment(config_path=None):
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
    image_formats = {extension.lower() for extension in config["dataset"]["image_format"]}
    logging_cfg = config["logging"]
    preview_dir = Path(config["experiment"]["results_dir"]) / "preprocessing"

    # -----------------------------
    # Optional preprocessing preview, controlled by logging.save_intermediate_images
    # -----------------------------
    # print("Performing preprocessing steps...")
    # for subfolder_name in class_subfolders:
    #     subfolder_path = os.path.join(dataset_dir, subfolder_name)
    #     for image_name in os.listdir(subfolder_path):
    #         if Path(image_name).suffix.lower() not in image_formats:
    #             continue

    #         image_path = os.path.join(subfolder_path, image_name)

    #         image = cv2.imread(image_path)
    #         if image is None:
    #             raise ValueError(f"Failed to load image: {image_path}")

    #         processed = _preprocess_from_config(image, config)

    #         if logging_cfg["save_intermediate_images"]:
    #             resized_original = preprocess_image(
    #                 image,
    #                 resize=(
    #                     tuple(config["preprocessing"]["resize"]["target_size"])
    #                     if config["preprocessing"]["resize"]["enabled"]
    #                     else None
    #                 ),
    #                 grayscale=config["dataset"]["grayscale"],
    #             )
    #             output_path = preview_dir / subfolder_name / image_name
    #             output_path.parent.mkdir(parents=True, exist_ok=True)
    #             cv2.imwrite(str(output_path), cv2.hconcat([resized_original, processed]))
    #             cv2.imwrite(str(output_path), processed)
              
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
    # print("Extracting phase congruency features...")
    # pc.extract_features(dataset_dir, class_subfolders, features_dir, save_images=True)
    
    
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
    combined = pd.read_csv(os.path.join(Path(features_dir), Path("Phase_congruency_features.csv")))

    # -----------------------------
    # Run classifier
    # -----------------------------
    run_svm(combined)
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
    run_experiment()
