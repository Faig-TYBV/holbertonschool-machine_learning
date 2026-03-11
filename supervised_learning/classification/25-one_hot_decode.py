#!/usr/bin/env python3
"""One hot decode class"""


import numpy as np


def one_hot_decode(one_hot):
    '''Converts a one-hot matrix into a numeric label vector'''

    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
