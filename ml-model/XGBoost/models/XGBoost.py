from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, RandomizedSearchCV)
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

all_y_pred = []
all_y_prob = []
all_y_test = []

xgb_base = XGBClassifier(
  objective = "binary:logistic",
  random_state = 42,
  eval_metric = "logloss"
)

"""
param_grid = {
  "n_estimators": [100, 200, 300, 500],
  "max_depth": [3, 4, 5, 6, 8],
  "learning_rate": [0.01, 0.03, 0.05, 0.1],

  "subsample": [0.6, 0.8, 1.0],
  "colsample_bytree": [0.6, 0.8, 1.0],

  "min_child_weight": [1, 3, 5, 7],

  "gamma": [0, 0.1, 0.3, 1],

  "reg_alpha": [0, 0.01, 0.1],
  "reg_lambda": [1, 1.5, 2]
}


search_param = RandomizedSearchCV(
  estimator= xgb_base,
  param_distributions= param_grid,
  n_iter= 200,
  scoring= "roc_auc",
  cv = list(gkf.split(X, y, groups)),
  verbose= 2,
  n_jobs= -1,
  random_state= 42
)
"""

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
    max_depth = 4,
    learning_rate = 0.03,
    subsample = 0.6,
    colsample_bytree = 0.6,
    random_state = 42,
    eval_metric = "logloss",
    reg_lambda = 1,
    reg_alpha = 0,
    min_child_weight = 3,
    gamma = 0.1
  )

  xg_model.fit(X_train, y_train)

  y_pred = xg_model.predict(X_test)

  y_prob = xg_model.predict_proba(X_test)[:, 1]
  
  all_y_test.extend(y_test)
  all_y_pred.extend(y_pred)
  all_y_prob.extend(y_prob)

"""
print("\nStarting Hyperparameter Search...\n")

search_param.fit(
  X, y, groups=groups
)

print("Best ROC AUC : ",search_param.best_score_)

print("Best Parameters:")

print(search_param.best_params_)

"""

print("ROC AUC : ", roc_auc_score(all_y_test, all_y_prob))

print("Accuracy : ", accuracy_score(all_y_test, all_y_pred))

print(classification_report(all_y_test, all_y_pred))

print(confusion_matrix(all_y_test, all_y_pred))

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": xg_model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print(importance[importance.importance > 0])
