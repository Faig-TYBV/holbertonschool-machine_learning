#!/usr/bin/env python3
"""6. Momentum
"""


import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the Gradient Descent with Momentum optimization algorithm.
    Args:
        alpha: The learning rate.
        beta1: The momentum weight.
    Returns:
        The optimizer object.
    """

    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
