#!/usr/bin/env python3
"""2. Shuffle Data
"""


import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way."""

    m = X.shape[0]
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    return shuffled_X, shuffled_Y
