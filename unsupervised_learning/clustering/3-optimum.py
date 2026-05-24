#!/usr/bin/env python3
"""Module for finding the optimum number of clusters by variance."""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Test for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        kmin (int): Minimum number of clusters to check (inclusive).
        kmax (int): Maximum number of clusters to check (inclusive).
        iterations (int): Maximum number of iterations for K-means.

    Returns:
        tuple: (results, d_vars) where results is a list of K-means outputs
               for each cluster size and d_vars is a list of variance
               differences from the smallest cluster size,
               or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
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

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        variances.append(variance(X, C))

    baseline = variances[0]
    d_vars = [baseline - v for v in variances]

    return results, d_vars
