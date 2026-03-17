#!/usr/bin/env python3
'''9-model.py
    - Save and load an entire model with Keras
'''


import tensorflow.keras as K


def save_model(network, filename):
    '''save_model - saves an entire model to a file'''

    network.save(filename)
    return None


def load_model(filename):
    '''load_model - loads an entire model from a file'''

    return K.models.load_model(filename)
