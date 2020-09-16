import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10, fashion_mnist, imdb, mnist
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_famnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    X = np.concatenate([x_train, x_test])  # Combine train/test to make new train/test/validate later on
    y = np.concatenate([y_train, y_test])

    X = X.reshape(-1, 28, 28, 1)
    X = X / 255
    y = y.reshape(y.shape[0], )

    return X, y


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.concatenate([x_train, x_test])  # Combine train/test to make new train/test/validate later on
    y = np.concatenate([y_train, y_test])

    X = X.reshape(-1, 28, 28, 1)
    X = X / 255
    y = y.reshape(y.shape[0], )

    return X, y


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    X = np.concatenate([x_train, x_test])  # Combine train/test to make new train/test/validate later on
    y = np.concatenate([y_train, y_test])

    X = X.reshape(-1, 32, 32, 3)
    X = X / 255
    y = y.reshape(y.shape[0], )

    return X, y


def load_imdb(config=(5_000, 500)):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=config[0])
    X = np.concatenate([x_train, x_test])  # Combine train/test to make new train/test/validate later on
    y = np.concatenate([y_train, y_test])

    X = pad_sequences(X, maxlen=config[1])

    return X, y


def load_creditcard(fp="./data/creditcard.csv"):
    """
    Loads the creditcard fraud dataset.
    Source: https://www.kaggle.com/mlg-ulb/creditcardfraud
    """
    X = pd.read_csv(fp)  # Directly converted to float64

    y = X["Class"]  # 1: Fraud/Minority, 0: No fraud/Majority
    X.drop(columns=["Time", "Class"], inplace=True)  # Dropping `Time` since future data for the model could have another epoch

    return X.values, y.values  # Numpy arrays


def load_data(data_source, imb_rate, min_class, maj_class, seed=None, normalization=False):
    """
    Loads data from the `data_source`. Imbalances the data and divides the data into train, test and validation sets.
    The imbalance rate of each individual dataset is the same as the `imb_rate`.
    """
    if data_source == "famnist":
        X, y = load_famnist()
    elif data_source == "mnist":
        X, y = load_mnist()
    elif data_source == "cifar10":
        X, y = load_cifar10()
    elif data_source == "credit":
        X, y = load_creditcard()
    elif data_source == "imdb":
        X, y = load_imdb()
    else:
        raise ValueError("No valid `data_source`.")

    X, y = get_imb_data(X, y, imb_rate, min_class, maj_class)  # Imbalance the data

    # 60 / 20 / 20 for train / test / validate
    # stratify=y to ensure class balance is kept
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=seed, stratify=y_test)

    if data_source == "credit" and normalization:
        # Normalize data. This does not happen in load_creditcard to prevent train/test/val leakage
        # Other data sources are already normalized. RGB values are always in range 0 to 255.
        mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
        for X in (X_train, X_test, X_val):  # Normalize to Z-score
            X -= mean
            X /= std

    p_data, p_train, p_test, p_val = [((y == 1).sum(), (y == 1).sum() / (y == 0).sum()) for y in [y, y_train, y_test, y_val]]
    print(f"Imbalance ratio `p`:\n"
          f"\tdataset:    n={p_data[0]}, p={p_data[1]:.6f}\n"
          f"\ttrain:      n={p_train[0]}, p={p_train[1]:.6f}\n"
          f"\ttest:       n={p_test[0]}, p={p_test[1]:.6f}\n"
          f"\tvalidation: n={p_val[0]}, p={p_val[1]:.6f}")

    return X_train, y_train, X_test, y_test, X_val, y_val


def get_imb_data(X, y, imb_rate, min_class, maj_class):
    """
    Split data in minority and majority, only values in {min_class, maj_class} will be kept.
    Decrease minority rows to match the imbalance rate.

    Note: Data will not be shuffled
    """
    X_min, y_min, X_maj, y_maj = [], [], [], []

    for i, value in enumerate(y):
        if value in min_class:
            X_min.append(X[i])
            y_min.append(1)

        if value in maj_class:
            X_maj.append(X[i])
            y_maj.append(0)

    min_len = int(len(y_maj) * imb_rate)

    # Keep all majority rows, decrease minority rows to match `imb_rate`
    X_imb = np.array(X_maj + X_min[:min_len])  # `min_len` can be more than the number of minority rows
    y_imb = np.array(y_maj + y_min[:min_len])
    return X_imb, y_imb


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, X_val, y_val = load_data("credit", 0.01, [1], [0], seed=42)
    print([i.shape for i in [X_train, y_train, X_test, y_test, X_val, y_val]])
