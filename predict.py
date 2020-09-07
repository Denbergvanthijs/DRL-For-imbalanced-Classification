import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import confusion_matrix

from get_data import get_imb_data, load_data


def make_predictions(model, X_test):
    """Input a trained model and X data, returns predictions on minority (0) or majority (1)."""
    return [np.argmax(x) for x in model.predict(X_test)]


def plot_conf_matrix(y_true, y_pred):
    """Plots confusion matrix. True labels on x-axis. Predicted labels on y-axis."""
    TP, FN, FP, TN = confusion_matrix(y_true, y_pred).ravel()  # Minority: positive, Majority: negative
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))  # Matthews Corr. Coefficient

    ticklabels = ["Minority", "Majority"]
    sns.heatmap(((TP, FN), (FP, TN)), annot=True, fmt="_d", cmap="viridis", xticklabels=ticklabels, yticklabels=ticklabels)
    plt.title(f"Confusion matrix\nMCC: {MCC:.6f}")
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
    X_train, y_train, X_test, y_test = get_imb_data(X_train, y_train, X_test, y_test, imb_rate, min_class, maj_class)

    model = load_model(fp_model)
    y_pred = make_predictions(model=model, data=X_test)
    plot_conf_matrix(y_test, y_pred)

    # plt.imshow(X_test[0], cmap="Greys")  # Show first image
    # plt.show()
