#!/usr/bin/env python3
"""L2 regularization cost function
"""


import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.
    Args:
        cost: tensor containing the cost of the network without
        L2 regularization
        model: a Keras model that includes layers with L2 regularization
    Returns:
        A tensor containing the total cost accounting for L2 regularization
    """

    # model.losses is a list of tensors (one per layer with a regularizer).
    # We sum these into a single scalar regularization term.
    l2_losses = tf.stack(model.losses)
    # Add the base cost to every layer's regularization term
    return cost + l2_losses
