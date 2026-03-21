#!/usr/bin/env python3
"""8. RMSProp
"""


import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm.
    Args:
        alpha: The learning rate.
        beta2: The RMSProp weight.
        epsilon: A small number to avoid division by zero.
    Returns:
        The optimizer object.
    """

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            rho=beta2, epsilon=epsilon)
    return optimizer
