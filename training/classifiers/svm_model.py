"""Train and evaluate SVM classifiers."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from training.feature_tables import classifier_inputs

def svm_definition():
  """Return a fresh estimator and the full Chong parameter grid."""
  return SVC(), {'C': [10**i for i in range(-5, 5)],
                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                 'degree': [2, 3, 4, 5],
                 'gamma': ['auto', 'scale']}

def run_svm(df):
  # Split the dataset into training and testing sets
  X, y = classifier_inputs(df)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  # Convert y_train and y_test to Pandas Series
  y_train = pd.Series(y_train)
  y_test = pd.Series(y_test)

  # Print number of training and testing images
  print("Number of training images: ", len(X_train))
  print("Number of testing images: ", len(X_test))

  # Print number of images for each class in training and testing sets
  print("Number of images for each class in training set:\n", y_train.value_counts())
  print("Number of images for each class in testing set:\n", y_test.value_counts())

  # Define the SVM model
  svm, param_grid = svm_definition()

  # Define the hyperparameters to optimize using GridSearchCV
  # Define the GridSearchCV object and fit it to the training data
  # cv=5 is 5-fold cross-validation
  grid_search = GridSearchCV(svm, param_grid, cv=10, n_jobs=-1, verbose=3, refit=True)

  # Fit the GridSearchCV object to the training data and get the results
  history = grid_search.fit(X_train, y_train)

  # Get the index of the best mean test score
  best_idx = np.argmax(history.cv_results_['mean_test_score'])

  # Print the best hyperparameters and the corresponding accuracy score
  print("Best hyperparameters: ", grid_search.best_params_)
  print("Accuracy score: ", grid_search.best_score_)
  print("Iteration: ", best_idx + 1)

  # Plot the number of iterations, accuracy and losses
  plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.plot(history.cv_results_['rank_test_score'], history.cv_results_['mean_test_score'], 'o-', label='Test accuracy')
  plt.xlabel('Iteration')
  plt.ylabel('Accuracy')
  plt.legend(loc='best')
  plt.subplot(1, 2, 2)
  plt.plot(history.cv_results_['rank_test_score'], history.cv_results_['mean_fit_time'], 'o-', label='Fit time')
  plt.plot(history.cv_results_['rank_test_score'], history.cv_results_['mean_score_time'], 'o-', label='Score time')
  plt.xlabel('Iteration')
  plt.ylabel('Time (s)')
  plt.legend(loc='best')
  plt.tight_layout()
  plt.show()

  # Predict the labels of the testing set using the optimized model
  y_pred = grid_search.predict(X_test)

  # Compute the accuracy score and confusion matrix on the testing set
  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix_test = confusion_matrix(y_test, y_pred)

  # Print the accuracy score and confusion matrix on the testing set
  print("Accuracy score on the testing set: ", accuracy)
  print("Confusion matrix on the testing set: ")
  print(conf_matrix_test)

  # Compute the accuracy score and confusion matrix on the training set
  y_pred_train = grid_search.predict(X_train)
  accuracy_train = accuracy_score(y_train, y_pred_train)
  conf_matrix_train = confusion_matrix(y_train, y_pred_train)

  # Print the accuracy score and confusion matrix on the training set
  print("Accuracy score on the training set: ", accuracy_train)
  print("Confusion matrix on the training set: ")
  print(conf_matrix_train)

  # Plot the confusion matrix for the testing set
  fig, ax = plt.subplots()
  im = ax.imshow(conf_matrix_test, interpolation='nearest', cmap=plt.cm.Blues)
  ax.figure.colorbar(im, ax=ax)

  # Set tick labels
  classes = np.unique(y)
  ax.set(xticks=np.arange(conf_matrix_test.shape[1]),
        yticks=np.arange(conf_matrix_test.shape[0]),
        xticklabels=classes, yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label')

  # Rotate the tick labels and set their alignment
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  # Loop over data dimensions and create text annotations
  thresh = conf_matrix_test.max() / 2.
  for i in range(conf_matrix_test.shape[0]):
      for j in range(conf_matrix_test.shape[1]):
          ax.text(j, i, format(conf_matrix_test[i, j], 'd'),
                  ha="center", va="center",
                  color="white" if conf_matrix_test[i, j] > thresh else "black")

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

  from sklearn.model_selection import cross_val_score

  # Initialize SVM with the best hyperparameters from GridSearchCV
  svm_best = SVC(**grid_search.best_params_)

  # Compute 5-fold cross-validation scores on the training set
  cv_scores = cross_val_score(svm_best, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)

  # Print cross-validation results
  print("Cross-validation scores:", cv_scores)
  print("Mean cross-validation accuracy: {:.4f}".format(cv_scores.mean()))
  print("Standard deviation of cross-validation accuracy: {:.4f}".format(cv_scores.std()))
