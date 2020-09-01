from keras.layers import Activation, Conv2D, Dense, Embedding, Flatten, MaxPooling2D
from keras.models import Sequential


def get_text_model(input_shape, output):
    top_words, max_words = input_shape
    model = Sequential()
    model.add(Embedding(top_words, 128, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dense(output))
    return model


def get_image_model(in_shape, output):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='Same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), padding='Same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(output))
    return model
