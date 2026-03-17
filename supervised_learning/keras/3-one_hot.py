#!/usr/bin/env python3
'''3-one_hot.py
    - One-hot encode data with Keras
'''


import tensorflow.keras as K


def one_hot(labels, classes=None):
    '''one_hot - converts a label vector into a one-hot matrix'''

    return K.utils.to_categorical(labels, num_classes=classes)
