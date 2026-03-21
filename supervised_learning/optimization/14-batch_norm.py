#!/usr/bin/env python3
"""14. Batch Normalization
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network.
    """
    # Use the specific initializer requested
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    # Create the Dense layer (remember use_bias=False as BN has beta)
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init,
                                  use_bias=False)
    z = dense(prev)
    # Create Gamma and Beta initializers as requested
    gamma_init = tf.keras.initializers.Constant(1)
    beta_init = tf.keras.initializers.Constant(0)
    # Setup Batch Normalization
    # Note: Ensure epsilon is exactly 1e-7 as requested
    bn = tf.keras.layers.BatchNormalization(
        gamma_initializer=gamma_init,
        beta_initializer=beta_init,
        epsilon=1e-7
    )
    # Apply normalization and then activation
    return activation(bn(z))
