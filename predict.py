import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from data_pre import get_imb_data, load_data

model = load_model("./models/test.h5")

imb_rate = 0.04  # Imbalance rate
min_class = [2]  # Minority classes, must be same as trained model
maj_class = [3]  # Majority classes, must be same as trained model

x_train, y_train, x_test, y_test = load_data("mnist")  # Load data to predict
# Remove classes ∉ {min_class, maj_class}, imbalance the datasrt
x_train, y_train, x_test, y_test = get_imb_data(x_train, y_train, x_test, y_test, imb_rate, min_class, maj_class)


print(f"y_true: {y_test[:20].tolist()}\ny_pred: {[np.argmax(x) for x in model.predict(x_test[:20])]}")  # 0: Minority, 1: Majority

plt.imshow(x_test[0], cmap='Greys')  # Show first image
plt.show()
