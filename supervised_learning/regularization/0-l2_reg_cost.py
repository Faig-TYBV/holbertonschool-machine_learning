#!/usr/bin/env python3
"""L2 regularization cost function
"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    Args:
        cost: cost of the network without L2 regularization
        lambtha: regularization parameter
        weights: dict of weights and biases (numpy.ndarrays)
        L: number of layers in the network
        m: number of data points used
    Returns:
        The cost of the network accounting for L2 regularization
    """

    l2_term = 0
    # Iterate through layers to sum the squared weights
    for i in range(1, L + 1):
        # We only regularize the weights 'W', not the biases 'b'
        l2_term += np.linalg.norm(weights['W' + str(i)])**2
    # Add the penalty to the original cost
    l2_cost = cost + (lambtha / (2 * m)) * l2_term
    return l2_cost
