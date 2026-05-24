#!/usr/bin/env python3
"""Module for performing Expectation Maximization for a GMM."""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """Perform the expectation maximization algorithm for a GMM.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.
        iterations (int): Maximum number of iterations.
        tol (float): Log likelihood tolerance for early stopping.
        verbose (bool): Whether to print log likelihood progress.

    Returns:
        tuple: (pi, m, S, g, l) where pi is a numpy.ndarray of shape (k,)
               containing the priors, m is a numpy.ndarray of shape (k, d)
               containing the centroid means, S is a numpy.ndarray of shape
               (k, d, d) containing the covariance matrices, g is a
               numpy.ndarray of shape (k, n) containing the posterior
               probabilities, and l is the log likelihood,
               or (None, None, None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    l_prev = 0
    g, l = None, None

    for i in range(iterations):
        g, l = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(l, 5)))

        if i > 0 and abs(l - l_prev) <= tol:
            break

        l_prev = l
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i, round(l, 5)))

    return pi, m, S, g, l
