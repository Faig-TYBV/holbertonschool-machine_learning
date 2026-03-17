#!/usr/bin/env python3
'''5-train.py
    - Train a model with Keras
'''


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    '''train_model - trains a model using mini-batch gradient descent'''

    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, validation_data=validation_data,
                       verbose=verbose, shuffle=shuffle)
