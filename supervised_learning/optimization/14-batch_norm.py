#!/usr/bin/env python3
"""14. Batch Normalization
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network.
    """
    # 1. Initialize the Base Dense Layer
    # Use VarianceScaling with fan_avg as requested
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    # Calculate the linear output: z = Wx + b
    z = dense(prev)
    # 2. Setup the Batch Normalization Layer
    # Use Constant initializers for gamma (1) and beta (0)
    # Use epsilon of 1e-7 exactly
    batch_norm = tf.keras.layers.BatchNormalization(
        gamma_initializer=tf.keras.initializers.Constant(1),
        beta_initializer=tf.keras.initializers.Constant(0),
        epsilon=1e-7
    )
    # Normalize the linear output
    z_norm = batch_norm(z)
    # 3. Apply the activation function
    return activation(z_norm)
