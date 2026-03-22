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
    reg_losses = tf.add_n(model.losses)
    # By adding the scalar 'reg_losses' to the 'cost' tensor,
    # TensorFlow broadcasts the addition so the penalty is applied
    # to every element in the tensor, preserving the shape (3,).
    return cost + reg_losses
