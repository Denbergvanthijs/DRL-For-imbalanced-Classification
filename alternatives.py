from sklearn.linear_model import LogisticRegression, SGDClassifier

from get_data import load_data
from utils import plot_conf_matrix
from xgboost import XGBClassifier

# Helpfull resources:
# https://xgboost.readthedocs.io/en/latest//parameter.html
# https://stats.stackexchange.com/questions/233248/max-delta-step-in-xgboost

imb_rate = 0.01  # Imbalance rate
min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes
datasource = "credit"  # The dataset to be selected

# Remove classes ∉ {min_class, maj_class}, imbalance the dataset
X_train, y_train, X_test, y_test, X_val, y_val = load_data(datasource, imb_rate, min_class, maj_class, print_stats=False)  # Load all data
scale = (y_train.shape[0] - y_train.sum()) // y_train.sum()  # Proportion of majority to minority rows

# XGBoost; Based on: https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-numpy.py
model = XGBClassifier(objective="binary:logitraw", scale_pos_weight=scale, eval_metric="aucpr", max_delta_step=1)
TP = [81, 82, 85, 83, 81]
TN = [56856, 56858, 56855, 56852, 56857]
FP = [7, 5, 8, 11, 6]  # 7; 2.0
FN = [17, 16, 13, 15, 17]  # 15; 1.5

# Logistic Regression; Used alot on the Credit Card Fraud Kaggle Competition
# Increase class_weight for less FN at the cost of alot FP
model = LogisticRegression(C=0.01, max_iter=1_000, class_weight={1: scale // 24})
TP = [77, 77, 83, 81, 77]
TN = [56816, 56827, 56815, 56817, 56822]
FP = [47, 36, 48, 46, 41]  # 43; 4.5
FN = [21, 21, 15, 17, 21]  # 19; 2.5

# StogasticGradientDescent; As recommended by https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
model = SGDClassifier(loss="log", alpha=0.001, max_iter=10_000, early_stopping=True, class_weight={1: scale // 24})
TP = [69, 74, 78, 79, 63]
TN = [56833, 56849, 56821, 56821, 56837]
FP = [30, 14, 42, 42, 26]  # 30; 10.5
FN = [29, 24, 20, 19, 35]  # 25; 5.9

# General code for all models
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
stats = plot_conf_matrix(y_test, y_pred)
