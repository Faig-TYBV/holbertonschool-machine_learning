#!/usr/bin/env python3
"""dropout create layer function
"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.
    Args:
        prev:       tensor - output of the previous layer
        n:          int - number of nodes in the new layer
        activation: activation function for the new layer
        keep_prob:  float - probability that a node will be kept
        training:   bool - whether the model is in training mode
    Returns:
        tensor - output of the new layer with dropout applied
    """

    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )
    A = layer(prev)
    A = tf.keras.layers.Dropout(rate=1 - keep_prob)(A, training=training)
    return A
