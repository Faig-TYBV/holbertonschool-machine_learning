#!/usr/bin/env python3
"""3. Mini-Batch
"""


import numpy as np


def create_mini_batches(X, Y, batch_size):
    """Creates mini-batches from the given data."""

    shuffle_data = __import__('2-shuffle_data').shuffle_data
    X, Y = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []
    for i in range(0, m, batch_size):
        mini_batch_X = X[i:i + batch_size]
        mini_batch_Y = Y[i:i + batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    return mini_batches
