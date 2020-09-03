import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from data_pre import get_imb_data, load_data


def make_predictions(model, data):
    """Input a trained model and X data, returns predictions on minority (0) or majority (1)."""
    return [np.argmax(x) for x in model.predict(data)]


if __name__ == "__main__":
    imb_rate = 0.0005  # Imbalance rate
    min_class = [2]  # Minority classes, must be same as trained model
    maj_class = [0, 1, 3, 4, 5, 6, 7, 8, 9]  # Majority classes, must be same as trained model

    x_train, y_train, x_test, y_test = load_data("mnist")  # Load data to predict
    # Remove classes ∉ {min_class, maj_class}, imbalance the dataset
    x_train, y_train, x_test, y_test = get_imb_data(x_train, y_train, x_test, y_test, imb_rate, min_class, maj_class)

    model = load_model("./models/mnistMin2MajAllOptimised.h5")
    y_pred = make_predictions(model=model, data=x_test[:20])
    print(f"y_true: {y_test[:20].tolist()}\ny_pred: {y_pred}")

    plt.imshow(x_test[0], cmap="Greys")  # Show first image
    plt.show()
