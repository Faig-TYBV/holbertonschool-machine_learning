#!/usr/bin/env python3
"""Module for calculating symmetric P affinities for t-SNE."""

import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        tol: maximum tolerance allowed for entropy difference
        perplexity: desired perplexity for all Gaussian distributions

    Returns:
        P: numpy.ndarray of shape (n, n) containing symmetric P affinities
    """

    D, P, betas, H = P_init(X, perplexity)

    n = X.shape[0]
    target_H = np.log2(perplexity)

    for i in range(n):
        beta_min = None
        beta_max = None

        Di = np.concatenate((D[i, :i], D[i, i + 1:]))

        Hi, Pi = HP(Di, betas[i])

        while np.abs(Hi - target_H) > tol:
            if Hi > target_H:
                beta_min = betas[i].copy()

                if beta_max is None:
                    betas[i] *= 2
                else:
                    betas[i] = (betas[i] + beta_max) / 2

            else:
                beta_max = betas[i].copy()

                if beta_min is None:
                    betas[i] /= 2
                else:
                    betas[i] = (betas[i] + beta_min) / 2

            Hi, Pi = HP(Di, betas[i])

        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

    P = (P + P.T) / (2 * n)

    return P
