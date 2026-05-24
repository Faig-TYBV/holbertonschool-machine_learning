#!/usr/bin/env python3
"""Module for performing K-means clustering on a dataset."""

import numpy as np


def initialize(X, k):
    """Initialize cluster centroids using a multivariate uniform distribution.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: Initialized centroids of shape (k, d), or None.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None

    low = X.min(axis=0)
    high = X.max(axis=0)

    return np.random.uniform(low=low, high=high, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """Perform K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.
        iterations (int): Maximum number of iterations.

    Returns:
        tuple: (C, clss) or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    C = initialize(X, k)
    if C is None:
        return None, None

    low = X.min(axis=0)
    high = X.max(axis=0)
    clss = None

    for _ in range(iterations):
        diffs = X[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        clss_new = np.argmin(dists, axis=1)

        # Update centroids BEFORE convergence check so the returned
        # C is always derived from the current (stable) assignments.
        C_new = np.zeros_like(C)
        for j in range(k):
            mask = clss_new == j
            if not mask.any():
                C_new[j] = np.random.uniform(low=low, high=high)
            else:
                C_new[j] = X[mask].mean(axis=0)

        if clss is not None and np.array_equal(clss_new, clss):
            return C_new, clss_new

        clss = clss_new
        C = C_new

    clss = np.argmin(
        np.linalg.norm(
            X[:, np.newaxis, :] - C[np.newaxis, :, :], axis=2
        ),
        axis=1
    )
    return C, clss
