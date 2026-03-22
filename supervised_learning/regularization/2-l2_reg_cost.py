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

    # model.losses contains the regularization penalties for all layers
    # We sum them up and add them to the original cost
    reg_losses = tf.add_n(model.losses)
    return cost + reg_losses
