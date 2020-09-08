import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

from get_data import get_imb_data, load_data
from ICMDP_Env import ClassifyEnv


def make_predictions(model, X_test):
    """Input a trained model and X data, returns predictions on minority (0) or majority (1)."""
    return [np.argmax(x) for x in model.predict(X_test)]


def plot_conf_matrix(y_true, y_pred):
    """Plots confusion matrix. True labels on x-axis. Predicted labels on y-axis."""
    info = ClassifyEnv.metrics(y_true, y_pred)
    TP, FN, FP, TN = confusion_matrix(y_true, y_pred).ravel()  # Minority: positive, Majority: negative
    ticklabels = ("Minority", "Majority")

    print(classification_report(y_true, y_pred, target_names=ticklabels))
    print(f"TP: {TP} TN: {TN} FP: {FP} FN: {FN}")
    print("".join([f"{k}: {v:.6f} " for k, v in info.items()]))

    sns.heatmap(((TP, FN), (FP, TN)), annot=True, fmt="_d", cmap="viridis", xticklabels=ticklabels, yticklabels=ticklabels)
    plt.title(f"Confusion matrix\nMCC: {info['MCC']:.6f}")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()


if __name__ == "__main__":
    # TODO: Determine why CPU is faster than GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1: Defaults to CPU, 0: GPU

    imb_rate = 0.0005  # Imbalance rate
    min_class = [1]  # Minority classes, must be same as trained model
    maj_class = [0]  # Majority classes, must be same as trained model
    datasource = "credit"  # The dataset to be selected
    fp_model = "./models/credit.h5"  # Filepath to the .h5-model

    X_train, y_train, X_test, y_test = load_data(datasource)  # Load all data
    # Remove classes ∉ {min_class, maj_class}, imbalance the dataset
    # Make sure the same seed is used as during training to ensure no data contamination
    X_train, y_train, X_test, y_test = get_imb_data(X_train, y_train, X_test, y_test, imb_rate, min_class, maj_class, seed=42)

    model = load_model(fp_model)
    y_pred = make_predictions(model, X_test)
    plot_conf_matrix(y_test, y_pred)  # Plot confidence matrix based on test dataset

    # y_baseline = np.ones(len(y_pred), dtype=int)  # Baseline, everything is Majority
    # plot_conf_matrix(y_test, y_baseline)

    # plt.imshow(X_test[0], cmap="Greys")  # Show first image
    # plt.show()
