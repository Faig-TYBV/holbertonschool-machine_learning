#!/usr/bin/env python3
"""
Performs K-means clustering
"""

import numpy as np

initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of iterations

    Returns:
        C, clss, or None, None on failure
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(k) is not int or k <= 0:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    C = initialize(X, k)

    if C is None:
        return None, None

    for _ in range(iterations):
        C_prev = np.copy(C)

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for j in range(k):
            if X[clss == j].shape[0] == 0:
                C[j] = initialize(X, 1)[0]
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        if np.array_equal(C, C_prev):
            break

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
