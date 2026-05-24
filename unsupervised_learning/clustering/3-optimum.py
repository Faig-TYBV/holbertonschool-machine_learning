#!/usr/bin/env python3
"""
Determines the optimum number of clusters by variance
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: minimum number of clusters to check, inclusive
        kmax: maximum number of clusters to check, inclusive
        iterations: maximum number of iterations for K-means

    Returns:
        results: list containing the outputs of K-means for each cluster size
        d_vars: list containing the difference in variance from the smallest
                cluster size for each cluster size
        None, None on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None

    if kmax <= kmin:
        return None, None

    if kmax > X.shape[0]:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    variances = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        if C is None or clss is None:
            return None, None

        var = variance(X, C)

        if var is None:
            return None, None

        results.append((C, clss))
        variances.append(var)

    base_var = variances[0]

    for var in variances:
        d_vars.append(base_var - var)

    return results, d_vars
