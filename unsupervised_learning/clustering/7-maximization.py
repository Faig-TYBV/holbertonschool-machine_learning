#!/usr/bin/env python3
"""Module for the maximization step of the EM algorithm for a GMM."""

import numpy as np


def maximization(X, g):
    """Calculate the maximization step in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        g (numpy.ndarray): Posterior probabilities of shape (k, n).

    Returns:
        tuple: (pi, m, S) where pi is a numpy.ndarray of shape (k,) containing
               the updated priors, m is a numpy.ndarray of shape (k, d)
               containing the updated centroid means, and S is a numpy.ndarray
               of shape (k, d, d) containing the updated covariance matrices,
               or (None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    n_k = np.sum(g, axis=1)

    pi = n_k / n
    m = (g @ X) / n_k[:, np.newaxis]

    S = np.zeros((k, d, d))

    for j in range(k):
        diff = X - m[j]
        S[j] = (g[j, :, np.newaxis] * diff).T @ diff / n_k[j]

    return pi, m, S
