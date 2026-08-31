#!/usr/bin/env python3
"""
Policy Gradient Module for Reinforcement Learning.
"""

import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix using the softmax function.

    Args:
        matrix (numpy.ndarray): The state matrix (e.g., shape (1, 4)).
        weight (numpy.ndarray): The weights matrix (e.g., shape (4, 2)).

    Returns:
        numpy.ndarray: The computed policy probabilities.
    """
    # Calculate the dot product of state and weights
    z = np.dot(matrix, weight)

    # Apply softmax function to convert to probabilities
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
