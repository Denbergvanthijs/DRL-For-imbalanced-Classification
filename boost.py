from get_data import load_data
from predict import plot_conf_matrix
from xgboost import XGBClassifier

# Based on: https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-numpy.py
# Helpfull resource: https://xgboost.readthedocs.io/en/latest//parameter.html

imb_rate = 0.01  # Imbalance rate
min_class = [1]  # Minority classes, must be same as trained model
maj_class = [0]  # Majority classes, must be same as trained model
datasource = "credit"  # The dataset to be selected

# Remove classes ∉ {min_class, maj_class}, imbalance the dataset
# Make sure the same seed is used as during training to ensure no data contamination
X_train, y_train, X_test, y_test, X_val, y_val = load_data(datasource, imb_rate, min_class, maj_class)  # Load all data

scale = (y_train.shape[0] - y_train.sum()) // y_train.sum()  # Proportion of majority to minority rows

xgb = XGBClassifier(objective="binary:logitraw", scale_pos_weight=scale, eval_metric="aucpr")
xgb.fit(X_train, y_train)
print(xgb)  # Print parameters

y_pred = xgb.predict(X_test)
plot_conf_matrix(y_test, y_pred)
