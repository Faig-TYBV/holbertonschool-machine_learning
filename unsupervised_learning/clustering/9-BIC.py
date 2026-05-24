#!/usr/bin/env python3
"""Module for finding the best number of GMM clusters using BIC."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using the BIC.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        kmin (int): Minimum number of clusters to check (inclusive).
        kmax (int): Maximum number of clusters to check (inclusive).
        iterations (int): Maximum number of iterations for the EM algorithm.
        tol (float): Tolerance for the EM algorithm.
        verbose (bool): Whether to print EM algorithm information.

    Returns:
        tuple: (best_k, best_result, l, b) where best_k is the best value
               for k, best_result is a tuple (pi, m, S) for the best k,
               l is a numpy.ndarray of shape (kmax - kmin + 1,) of log
               likelihoods, and b is a numpy.ndarray of shape
               (kmax - kmin + 1,) of BIC values,
               or (None, None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmax <= kmin:
        return None, None, None, None
    if kmax > X.shape[0]:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    num_k = kmax - kmin + 1

    log_likelihoods = np.zeros(num_k)
    bic_values = np.zeros(num_k)
    results = []

    for i, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, lk = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None:
            return None, None, None, None

        results.append((pi, m, S))
        log_likelihoods[i] = lk

        # Parameters: k-1 priors + k*d means + k*d*(d+1)/2 covariance entries
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)
        bic_values[i] = p * np.log(n) - 2 * lk

    best_idx = np.argmin(bic_values)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, log_likelihoods, bic_values
