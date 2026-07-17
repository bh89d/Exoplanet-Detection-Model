from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupShuffleSplit, GroupKFold)
from sklearn.metrics import (
  classification_report,
  confusion_matrix,
  roc_auc_score,
  accuracy_score
)
from xgboost import XGBClassifier

features_df = pd.read_csv("data/ml/features/features.csv")

metadata_df = pd.read_csv("data/ml/features/metadata.csv")

groups = metadata_df["target_id"]

X = features_df.drop(columns=["label"])

y = features_df["label"]

gkf = GroupKFold(n_splits=5)

auc_scores = []

gss = GroupShuffleSplit(
  n_splits= 1,
  test_size= 0.2 ,
  random_state= 42
)

train_idx, test_idx = next(
  gss.split(
    X, y, groups
  )
)

for fold, (train_idx, test_idx) in enumerate(
  gkf.split(X, y, groups),
  start = 1
):

  X_train = X.iloc[train_idx]
  X_test = X.iloc[test_idx]

  y_train = y.iloc[train_idx]
  y_test = y.iloc[test_idx]

  xg_model = XGBClassifier(
    objective="binary:logistic",
    n_estimators = 300,
    max_depth = 6,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42,
    eval_metric = "logloss"
  )

  xg_model.fit(X_train, y_train)

  y_pred = xg_model.predict(X_test)

  y_prob = xg_model.predict_proba(X_test)[:, 1]
  
  auc = roc_auc_score(
    y_test, y_prob
  )
  
  auc_scores.append(auc)
  
  print(
    f"Fold {fold}: {auc:.4f}"
)

print("Mean ROC AUC : ", np.mean(auc_scores))

print("std ROC AUC : ", np.std(auc_scores))

"""

print("Accuracy : ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": xg_model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print(importance)

"""