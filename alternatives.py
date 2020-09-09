from sklearn.linear_model import LogisticRegression, SGDClassifier

from get_data import load_data
from predict import plot_conf_matrix
from xgboost import XGBClassifier

# Based on: https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-numpy.py
# Helpfull resource: https://xgboost.readthedocs.io/en/latest//parameter.html
# https://stats.stackexchange.com/questions/233248/max-delta-step-in-xgboost

imb_rate = 0.01  # Imbalance rate
min_class = [1]  # Minority classes, must be same as trained model
maj_class = [0]  # Majority classes, must be same as trained model
datasource = "credit"  # The dataset to be selected

# Remove classes ∉ {min_class, maj_class}, imbalance the dataset
# Make sure the same seed is used as during training to ensure no data contamination
X_train, y_train, X_test, y_test, X_val, y_val = load_data(datasource, imb_rate, min_class, maj_class)  # Load all data

scale = (y_train.shape[0] - y_train.sum()) // y_train.sum()  # Proportion of majority to minority rows

# XGBoost
model = XGBClassifier(objective="binary:logitraw", scale_pos_weight=scale, eval_metric="aucpr", max_delta_step=1)

# Logistic Regression
# Increase class_weight for less FN at the cost of alot FP
model = LogisticRegression(C=0.01, max_iter=1_000, class_weight={1: scale // 24})

# StogasticGradientDescent
model = SGDClassifier(loss="log", alpha=0.001, max_iter=10_000, early_stopping=True, class_weight={1: scale // 24})

# General code for all models
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plot_conf_matrix(y_test, y_pred)
