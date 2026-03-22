#!/usr/bin/env python3
"""L2 regularization gradient descent function
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization.
    Args:
        Y: one-hot numpy.ndarray (classes, m) containing correct labels
        weights: dict of weights and biases
        cache: dict of the outputs (activations) of each layer
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers
    """

    m = Y.shape[1]
    # Initial dZ for the output layer (Softmax + Cross-Entropy)
    # dZ = A[L] - Y
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        # Activation from the previous layer (A0 is the input X)
        A_prev = cache['A' + str(i - 1)]
        # Calculate gradients
        # dW = (1/m) * (dZ @ A_prev.T) + (lambtha/m) * W
        dW = (np.matmul(dZ, A_prev.T) / m)
        dW += (lambtha / m) * weights['W' + str(i)]
        db = np.sum(dZ, axis=1, keepdims=True) / m
        # Prepare dZ for the next layer (the previous layer in backprop)
        if i > 1:
            W = weights['W' + str(i)]
            A_prev = cache['A' + str(i - 1)]
            # Derivative of tanh: 1 - A^2
            dZ = np.matmul(W.T, dZ) * (1 - (A_prev ** 2))
        # Update weights and biases in place
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
