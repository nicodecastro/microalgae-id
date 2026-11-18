"""Train and evaluate k-nearest-neighbor classifiers."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from training.feature_tables import classifier_inputs
from training.splitting import legacy_splits

def knn_definition():
  """Return a fresh estimator and the full Chong parameter grid."""
  return KNeighborsClassifier(), {
      'n_neighbors': list(range(1, 42, 2)),
      'weights': ['uniform', 'distance'],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
      'p': [1, 2],
  }

def run_knn(df):
  # Split the dataset into training and testing sets
  X, y = classifier_inputs(df)
  knn, param_grid = knn_definition()
  train, test, inner = legacy_splits(df, y, knn, param_grid)
  X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

  # Convert y_train and y_test to Pandas Series
  y_train = pd.Series(y_train)
  y_test = pd.Series(y_test)

  # Print number of training and testing images
  print("Number of training images: ", len(X_train))
  print("Number of testing images: ", len(X_test))

  # Print number of images for each class in training and testing sets
  print("Number of images for each class in training set:\n", y_train.value_counts())
  print("Number of images for each class in testing set:\n", y_test.value_counts())

  # Perform grid search to find the best hyperparameters using k-fold cross validation
  grid_search = GridSearchCV(knn, param_grid, cv=inner, n_jobs=-1, verbose=3, refit=True)
  # TODO(augmentation): Generate image variants only from each fit's training
  # parents, then extract their features before fitting the scaler/model.
  # Keep validation/test originals unaugmented; variants inherit parent_image_id.
  # Never augment the holdout training partition before inner splitting.
  grid_search.fit(X_train, y_train)

  # Print the best hyperparameters
  print('Best hyperparameters:', grid_search.best_params_)

  # Initialize a K-NN classifier with the best hyperparameters
  knn = grid_search.best_estimator_

  # GridSearchCV already refitted the selected classifier on training data.

  # Evaluate the classifier on the testing data
  y_pred = knn.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print('Accuracy:', accuracy)

  # Compute the confusion matrix
  cm = confusion_matrix(y_test, y_pred)

  # Plot the confusion matrix
  fig, ax = plt.subplots()
  im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  ax.figure.colorbar(im, ax=ax)
  # Set tick labels
  classes = np.unique(y)
  ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label')
  # Rotate the tick labels and set their alignment
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")
  # Loop over data dimensions and create text annotations
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
          ax.text(j, i, format(cm[i, j], 'd'),
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()
  plt.show()

  # Advanced evaluation metrics
  from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
  precision = precision_score(y_test, y_pred, average='macro')
  recall = recall_score(y_test, y_pred, average='macro')
  f1 = f1_score(y_test, y_pred, average='macro')
  classification = classification_report(y_test, y_pred)
  print("Precision: ", precision)
  print("Recall: ", recall)
  print("F1 score: ", f1)
  print("Classification report: ")
  print(classification)

  # Summarize the selected candidate's grouped tuning scores without refitting.
  cv_scores = np.array([grid_search.cv_results_[f"split{i}_test_score"][grid_search.best_index_]
                        for i in range(len(inner))])
  print("Grouped tuning scores:", cv_scores)
  print("Mean grouped tuning accuracy:", cv_scores.mean())
  print("Standard deviation of grouped tuning accuracy:", cv_scores.std())
