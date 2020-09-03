import numpy as np
import pandas as pd
from keras.datasets import cifar10, fashion_mnist, imdb, mnist
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def load_famnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = y_train.reshape(y_train.shape[0], )
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test = y_test.reshape(y_test.shape[0], )
    x_train = x_train / 255.
    x_test = x_test / 255.

    return x_train, y_train, x_test, y_test


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = y_train.reshape(y_train.shape[0], )
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test = y_test.reshape(y_test.shape[0], )
    x_train = x_train / 255.
    x_test = x_test / 255.

    return x_train, y_train, x_test, y_test


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3)
    y_train = y_train.reshape(y_train.shape[0], )
    x_test = x_test.reshape(-1, 32, 32, 3)
    y_test = y_test.reshape(y_test.shape[0], )
    x_train = x_train / 255.
    x_test = x_test / 255.

    return x_train, y_train, x_test, y_test


def load_imdb():
    config = [5_000, 500]
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=config[0])

    x_train = pad_sequences(x_train, maxlen=config[1])
    x_test = pad_sequences(x_test, maxlen=config[1])

    return x_train, y_train, x_test, y_test


def load_creditcard(seed=42, fp="./data/creditcard.csv"):
    df = pd.read_csv(fp)  # Directly converted to float64

    df_y = df["Class"]
    df.drop(columns=["Time", "Class"], inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(df.values, df_y.values, test_size=0.2, random_state=seed)

    return x_train, y_train, x_test, y_test


def load_data(data_name):
    if data_name == "famnist":
        x_train, y_train, x_test, y_test = load_famnist()
    elif data_name == "mnist":
        x_train, y_train, x_test, y_test = load_mnist()
    elif data_name == "cifar10":
        x_train, y_train, x_test, y_test = load_cifar10()
    elif data_name == "credit":
        x_train, y_train, x_test, y_test = load_creditcard()
    else:
        x_train, y_train, x_test, y_test = load_imdb()

    return x_train, y_train, x_test, y_test


def get_imb_data(x_train, y_train, x_test, y_test, imb_rate, min_class, maj_class):
    maj_x_train = []
    maj_y_train = []
    min_x_train = []
    min_y_train = []

    for i in range(len(y_train)):
        if y_train[i] in min_class:
            min_x_train.append(x_train[i])
            min_y_train.append(0)

        if y_train[i] in maj_class:
            maj_x_train.append(x_train[i])
            maj_y_train.append(1)

    min_len = int(len(maj_y_train) * imb_rate)

    new_x_train = maj_x_train + min_x_train[:min_len]
    new_y_train = maj_y_train + min_y_train[:min_len]
    new_x_test = []
    new_y_test = []

    for i in range(len(y_test)):
        if y_test[i] in min_class:
            new_x_test.append(x_test[i])
            new_y_test.append(0)

        if y_test[i] in maj_class:
            new_x_test.append(x_test[i])
            new_y_test.append(1)

    new_x_train = np.array(new_x_train)
    new_y_train = np.array(new_y_train)
    new_x_test = np.array(new_x_test)
    new_y_test = np.array(new_y_test)

    idx = [i for i in range(len(new_y_train))]
    np.random.shuffle(idx)

    new_x_train = new_x_train[idx]
    new_y_train = new_y_train[idx]

    return new_x_train, new_y_train, new_x_test, new_y_test


if __name__ == "__main__":
    data = load_creditcard()
    print([i.shape for i in data])
