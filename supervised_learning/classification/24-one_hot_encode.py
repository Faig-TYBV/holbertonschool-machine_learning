#!usr/bin/end python3
"""One hot encode"""


import numpy as np


def one_hot_encode(Y, classes):
    '''Converts a numeric label vector into a one-hot matrix'''

    one_hot = np.zeros((classes, Y.shape[0]))
    one_hot[Y, np.arange(Y.shape[0])] = 1
    return one_hot
