#!/usr/bin/env python3
"""14. Batch Normalization
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.
    Args:
        prev: The activated output of the previous layer.
        n: The number of nodes in the layer to be created.
        activation: The activation function to be used on the output.  
    Returns:
        A tensor of the activated output for the layer.
    """
    # 1. Initialize the Base Dense Layer
    # We set use_bias=False because Batch Norm's 'beta' parameter
    # effectively handles the bias/offset.
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )
    # Calculate the linear output: z = Wx
    z = dense_layer(prev)
    # 2. Setup the Batch Normalization Layer
    # gamma (scale) initialized to 1s, beta (offset) initialized to 0s
    batch_norm = tf.keras.layers.BatchNormalization(
        beta_initializer='zeros',
        gamma_initializer='ones',
        epsilon=1e-7
    )
    # Normalize the linear output
    z_norm = batch_norm(z)
    # 3. Apply the Activation Function
    return activation(z_norm)
