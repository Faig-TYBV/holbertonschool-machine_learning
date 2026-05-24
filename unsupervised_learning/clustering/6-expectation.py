#!/usr/bin/env python3
"""Module for the expectation step of the EM algorithm for a GMM."""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculate the expectation step in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        pi (numpy.ndarray): Priors for each cluster of shape (k,).
        m (numpy.ndarray): Centroid means of shape (k, d).
        S (numpy.ndarray): Covariance matrices of shape (k, d, d).

    Returns:
        tuple: (g, l) where g is a numpy.ndarray of shape (k, n) containing
               the posterior probabilities and l is the total log likelihood,
               or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d):
        return None, None
    if S.shape != (k, d, d):
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    g = np.zeros((k, n))

    for j in range(k):
        g[j] = pi[j] * pdf(X, m[j], S[j])

    total = np.sum(g, axis=0)
    g = g / total

    log_likelihood = np.sum(np.log(total))

    return g, log_likelihood
