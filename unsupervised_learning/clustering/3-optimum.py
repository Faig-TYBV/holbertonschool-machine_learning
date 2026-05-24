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

    Returns:
        results: list of (C, clss) from kmeans for each k
        d_vars: list of variance differences from the smallest k
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None

    if kmax <= kmin or kmax > X.shape[0]:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        result = kmeans(X, k, iterations)

        if result is None:
            return None, None

        C, clss = result

        if C is None or clss is None:
            return None, None

        results.append((C, clss))

    base_var = variance(X, results[0][0])

    if base_var is None:
        return None, None

    for C, _ in results:
        var = variance(X, C)

        if var is None:
            return None, None

        d_vars.append(base_var - var)

    return results, d_vars
