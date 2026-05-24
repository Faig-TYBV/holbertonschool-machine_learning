#!/usr/bin/env python3
"""Module for initializing K-means cluster centroids."""

import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means clustering.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: Initialized centroids of shape (k, d), or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None

    low = X.min(axis=0)
    high = X.max(axis=0)

    return np.random.uniform(low=low, high=high, size=(k, X.shape[1]))
