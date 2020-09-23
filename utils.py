import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error)
from tensorflow.keras.models import load_model

from get_data import load_data


def make_predictions(model, X_test) -> list:
    """Input a trained model and X data, returns predictions on minority (1) or majority (0)."""
    return [np.argmax(x) for x in model.predict(X_test)]


def calculate_metrics(y_true, y_pred) -> dict:
    """Calculate metrics for a given y_true and y_pred."""
    # Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()  # Minority: positive, Majority: negative

    # Source: https://en.wikipedia.org/wiki/Precision_and_recall
    precision = TP / (TP + FP)  # Positive Predictive Value
    recall = TP / (TP + FN)  # Sensitivity, True Positive Rate (TPR)
    specificity = TN / (TN + FP)  # Specificity, selectivity, True Negative Rate (TNR)

    G_mean = np.sqrt(recall * specificity)  # Geometric mean of recall and specificity, defined in paper
    F_measure = np.sqrt(recall * precision)  # F-measure of recall and precision, defined in paper
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))  # Matthews Corr. Coefficient
    MAE = mean_absolute_error(y_true, y_pred)  # Mean absolute error

    return {"gmean": G_mean, "fmeasure": F_measure, "MCC": MCC, "MAE": MAE, "precision": precision, "recall": recall,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def plot_conf_matrix(y_true, y_pred) -> dict:
    """
    Plots confusion matrix. True labels on x-axis. Predicted labels on y-axis.
    Returns stats from `calculate_metrics()`.
    """
    stats = calculate_metrics(y_true, y_pred)
    ticklabels = ("Minority", "Majority")

    print(classification_report(y_true, y_pred, target_names=ticklabels[::-1]))
    print("".join([f"{k}: {v:.6f} " for k, v in stats.items()]))

    sns.heatmap(((stats.get("TP"), stats.get("FN")), (stats.get("FP"), stats.get("TN"))),
                annot=True, fmt="_d", cmap="viridis", xticklabels=ticklabels, yticklabels=ticklabels)

    plt.title(f"Confusion matrix\nMCC: {stats['MCC']:.6f}")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()

    return stats


if __name__ == "__main__":
    # TODO: Determine why CPU is faster than GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1: Defaults to CPU, 0: GPU

    imb_rate = 0.01  # Imbalance rate
    min_class = [1]  # Minority classes, must be same as trained model
    maj_class = [0]  # Majority classes, must be same as trained model
    datasource = "credit"  # The dataset to be selected
    fp_model = "./models/credit.h5"  # Filepath to the .h5-model

    # Remove classes ∉ {min_class, maj_class}, imbalance the dataset
    # Make sure the same seed is used as during training to ensure no data contamination
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(datasource, imb_rate, min_class, maj_class)  # Load all data

    model = load_model(fp_model)
    y_pred = make_predictions(model, X_test)
    plot_conf_matrix(y_test, y_pred)  # Plot confidence matrix based on test dataset

    # y_baseline = np.ones(len(y_pred), dtype=int)  # Baseline, everything is Majority
    # plot_conf_matrix(y_val, y_baseline)

    # plt.imshow(X_test[0], cmap="Greys")  # Show first image
    # plt.show()
