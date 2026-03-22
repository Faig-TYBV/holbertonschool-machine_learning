#!/usr/bin/env python3
"""L3 regularization create layer function
"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow with L2 regularization.

    Args:
        prev: Tensor containing the output of the previous layer.
        n: Number of nodes the new layer should contain.
        activation: Activation function to be used on the layer.
        lambtha: L2 regularization parameter.

    Returns:
        The output of the new layer.
    """

    # Define the L2 regularizer with the given lambtha
    regularizer = tf.keras.regularizers.L2(l2=lambtha)
    # Initialize weights using He normal initialization (common practice
    #  for deep networks)
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode='fan_avg')
    # Create the Dense layer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer,
        kernel_initializer=initializer
    )
    return layer(prev)
