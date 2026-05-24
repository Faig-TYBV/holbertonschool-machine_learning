#!/usr/bin/env python3
"""Module for calculating the PDF of a Gaussian distribution."""

import numpy as np


def pdf(X, m, S):
    """Calculate the probability density function of a Gaussian distribution.

    Args:
        X (numpy.ndarray): Data points of shape (n, d).
        m (numpy.ndarray): Mean of the distribution of shape (d,).
        S (numpy.ndarray): Covariance matrix of shape (d, d).

    Returns:
        numpy.ndarray: PDF values of shape (n,), or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d:
        return None
    if S.shape != (d, d):
        return None

    det = np.linalg.det(S)
    if det <= 0:
        return None

    S_inv = np.linalg.inv(S)

    norm = 1.0 / (np.sqrt(((2 * np.pi) ** d) * det))

    diff = X - m
    exponent = -0.5 * np.sum(diff @ S_inv * diff, axis=1)

    P = norm * np.exp(exponent)

    return np.maximum(P, 1e-300)
