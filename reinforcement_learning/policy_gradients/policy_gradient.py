#!/usr/bin/env python3
"""
Policy Gradient Module for Reinforcement Learning.
"""

import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix using the softmax function.

    Args:
        matrix (numpy.ndarray): The state matrix.
        weight (numpy.ndarray): The weights matrix.

    Returns:
        numpy.ndarray: The computed policy probabilities.
    """
    z = np.dot(matrix, weight)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and
    a weight matrix.

    Args:
        state (numpy.ndarray): Matrix representing the current observation
            of the environment. (1D or 2D array)
        weight (numpy.ndarray): Matrix of random weights.

    Returns:
        tuple: (action, gradient)
            - action (int): The selected action based on the policy.
            - gradient (numpy.ndarray): The computed gradient.
    """
    # Ensure state is a 2D row vector (1, n_features)
    if state.ndim == 1:
        state = state.reshape(1, -1)

    # Get the policy probabilities for the given state
    probs = policy(state, weight)

    # Sample an action based on the probability distribution
    action = np.random.choice(len(probs[0]), p=probs[0])

    # Compute the gradient of the log probability
    # d_log(pi) = x * (y - pi)
    # where y is a one-hot encoded vector of the chosen action
    dsoftmax = np.zeros_like(probs)
    dsoftmax[0, action] = 1
    dsoftmax -= probs

    # Calculate the gradient: state^T * dsoftmax
    gradient = np.dot(state.T, dsoftmax)

    return action, gradient
