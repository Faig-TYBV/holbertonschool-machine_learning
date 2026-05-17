#!/usr/bin/env python3
"""Module for initializing variables for P affinities in t-SNE."""

import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate P affinities in t-SNE.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        perplexity: desired perplexity for the Gaussian distributions

    Returns:
        D: numpy.ndarray of shape (n, n) containing squared pairwise distances
        P: numpy.ndarray of shape (n, n) initialized to 0s
        betas: numpy.ndarray of shape (n, 1) initialized to 1s
        H: Shannon entropy for the given perplexity, base 2
    """

    n, d = X.shape

    sum_X = np.sum(np.square(X), axis=1)

    D = (
        sum_X.reshape((n, 1))
        + sum_X.reshape((1, n))
        - 2 * np.matmul(X, X.T)
    )

    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = np.log2(perplexity)

    return D, P, betas, H
