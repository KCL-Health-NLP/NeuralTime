
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


def train_network(train_data, train_labels):
    network = models.Sequential()
    network.add(layers.Dense(300, activation='relu', input_shape=(300,)))
    network.add(layers.Dense(200, activation='relu', ))
    network.add(layers.Dropout(0.5))
    network.add(layers.Dense(200, activation='relu'))
    network.add(layers.Dense(200, activation='relu'))
    network.add(layers.Dense(4, activation='softmax'))

    network.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    network.fit(train_data, train_labels, epochs=15, batch_size=1, shuffle='batch')
    return network
