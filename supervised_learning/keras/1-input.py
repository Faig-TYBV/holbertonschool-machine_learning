#!/usr/bin/env python3
'''1-input.py
    - Define the input layer for a neural network with Keras
'''


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''build_model - creates a sequential model with Keras'''

    model = K.Sequential()
    model.add(K.layers.Input(shape=(nx,)))
    for i in range(1, len(layers)):
        
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=K.
                                 regularizers.l2(lambtha)))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model