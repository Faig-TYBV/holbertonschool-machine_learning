#!/usr/bin/env python3
"""10. Adam
"""


import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm
    Args:
        alpha: is the learning rate
        beta1: is the weight used for the first moment
        beta2: is the weight used for the second moment
        epsilon: is a small number to avoid division by zero
    Returns:
        The Adam optimizer
    """

    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
