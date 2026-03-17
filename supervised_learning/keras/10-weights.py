#!/usr/bin/env python3
'''10-weights.py
    - Save and load model weights with Keras
'''


import tensorflow.keras as K


def save_weights(network, filename):
    '''save_weights - saves model weights to a file'''

    network.save_weights(filename)
    return None


def load_weights(network, filename):
    '''load_weights - loads model weights from a file'''

    network.load_weights(filename)
    return None
