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

    # model.losses is a list containing the regularization loss for each layer.
    # tf.add_n sums these individual layer penalties into a
    # single scalar tensor.
    l2_penalties = tf.add_n(model.losses)
    # By adding the scalar l2_penalties to the 'cost' tensor,
    # TensorFlow adds that scalar to every element in 'cost'.
    return cost + l2_penalties
