#!/usr/bin/env python3
"""Module for initializing Gaussian Mixture Model variables."""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize variables for a Gaussian Mixture Model.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.

    Returns:
        tuple: (pi, m, S) where pi is a numpy.ndarray of shape (k,) containing
               the priors, m is a numpy.ndarray of shape (k, d) containing the
               centroid means, and S is a numpy.ndarray of shape (k, d, d)
               containing the covariance matrices,
               or (None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None

    d = X.shape[1]

    pi = np.full((k,), 1 / k)

    m, clss = kmeans(X, k)
    if m is None:
        return None, None, None

    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
