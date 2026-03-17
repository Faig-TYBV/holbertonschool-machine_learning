#!/usr/bin/env python3
'''2-optimize.py
    - Optimize a model with Keras
'''


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    '''optimize_model - sets up Adam optimization for a keras model'''

    optimizer = K.optimizers.Adam(learning_rate=alpha,
                                  beta_1=beta1,
                                  beta_2=beta2)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
