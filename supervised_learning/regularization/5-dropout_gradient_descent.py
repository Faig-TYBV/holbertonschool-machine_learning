#!/usr/bin/env python3
"""dropout gradient descent function
"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates weights of a neural network with Dropout using gradient descent.
    Args:
        Y:         numpy.ndarray (classes, m) - correct labels
        weights:   dict - weights and biases of the network
        cache:     dict - outputs and dropout masks of each layer
        alpha:     float - learning rate
        keep_prob: float - probability that a node will be kept
        L:         int - number of layers
    """

    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y
    for layer in range(L, 0, -1):
        A_prev = cache[f'A{layer - 1}']
        W = weights[f'W{layer}']
        dW = (dZ @ A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = W.T @ dZ
        if layer > 1:
            dA_prev = dA_prev * cache[f'D{layer - 1}']
            dA_prev = dA_prev / keep_prob
            dZ = dA_prev * (1 - cache[f'A{layer - 1}'] ** 2)
        weights[f'W{layer}'] = weights[f'W{layer}'] - alpha * dW
        weights[f'b{layer}'] = weights[f'b{layer}'] - alpha * db
