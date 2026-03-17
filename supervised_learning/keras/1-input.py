#!/usr/bin/env python3
'''1-input.py
    - Define the input layer for a neural network with Keras
'''


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''build_model - creates a model with Keras Functional API'''

    # 1. Define the input layer explicitly
    inputs = K.Input(shape=(nx,))
    # 2. Chain the layers
    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)
        # Add dropout if it's not the last layer
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    # 3. Create the model by connecting inputs to outputs
    model = K.Model(inputs=inputs, outputs=x)
    return model
