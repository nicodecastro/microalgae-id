import numpy as np
from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  confusion_matrix,
  classification_report
)

def classification_metrics(y_true, y_pred, average="macro"):
  return {
      "accuracy": accuracy_score(y_true, y_pred),
      "precision": precision_score(y_true, y_pred, average=average),
      "recall": recall_score(y_true, y_pred, average=average),
      "f1": f1_score(y_true, y_pred, average=average),
  }

def compute_confusion(y_true, y_pred):
  return confusion_matrix(y_true, y_pred)
  
def summarize_cv(scores):
  return {
    "mean": np.mean(scores),
    "std": np.std(scores),
    "min": np.min(scores),
    "max": np.max(scores)
  }
    
def per_class_report(y_true, y_pred):
  return classification_report(y_true, y_pred, output_dict=True)