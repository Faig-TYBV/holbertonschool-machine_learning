#!/usr/bin/env python3
"""0. Constants
"""


def normalization_constants(X):
    """Calculates the normalization constants of a matrix:
      mean and standard deviation.
    """
    mean = X.mean(axis=0)
    stddev = X.std(axis=0)
    return mean, stddev
