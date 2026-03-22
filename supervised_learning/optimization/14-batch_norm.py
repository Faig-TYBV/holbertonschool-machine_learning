#!/usr/bin/env python3
"""14. Batch Normalization
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.
    Args:
        prev:       tensor - activated output of the previous layer
        n:          int - number of nodes in the layer
        activation: activation function for the output of the layer
    Returns:
        tensor - activated output of the batch normalized layer
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, kernel_initializer=initializer)
    Z = layer(prev)
    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')
    mean, variance = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma,
                                       variance_epsilon=1e-7)
    return activation(Z_norm)
