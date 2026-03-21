#!/usr/bin/env python3
"""13. Batch Normalization
"""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a neural network using batch
    normalization
    Args:
        Z: is a numpy.ndarray of shape (m, n) containing the unactivated
           output of the layer to be normalized
            m: is the number of data points
            n: is the number of features in Z
        gamma: is a numpy.ndarray of shape (1, n) containing the scales used
               for batch normalization
        beta: is a numpy.ndarray of shape (1, n) containing the offsets used
              for batch normalization
        epsilon: is a small number to avoid division by zero
    Returns:
        The normalized Z matrix
    """

    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_batch = gamma * Z_norm + beta
    return Z_batch
