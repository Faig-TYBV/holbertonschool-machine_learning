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

    # Sum all regularization losses present in the model
    # tf.add_n performs element-wise addition of a list of tensors
    l2_costs = tf.add_n(model.losses)
    # Adding the scalar l2_costs to the cost tensor will broadcast
    # the addition across each element in the cost tensor.
    return cost + l2_costs
