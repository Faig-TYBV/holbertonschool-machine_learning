#!/usr/bin/env python3
"""dropout forward propagation function
"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.
    Args:
        X:         numpy.ndarray (nx, m) - input data
        weights:   dict - weights and biases {'W1', 'b1', 'W2', 'b2', ...}
        L:         int  - number of layers
        keep_prob: float - probability that a node will be kept
    Returns:
        dict - outputs (A) and dropout masks (D) for each layer
    """

    cache = {}
    cache['A0'] = X
    for layer in range(1, L + 1):
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']
        A_prev = cache[f'A{layer - 1}']
        Z = W @ A_prev + b
        if layer == L:
            e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = e_Z / np.sum(e_Z, axis=0, keepdims=True)
            cache[f'A{layer}'] = A
        else:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            D = D.astype(int)
            A = (A * D) / keep_prob
            cache[f'D{layer}'] = D
            cache[f'A{layer}'] = A
    return cache
