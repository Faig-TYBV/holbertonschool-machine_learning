#!/usr/bin/env python3
'''2-identity_block.py'''


import tensorflow as tf


def identity_block(A_prev, filters):
    '''builds an identity block as described in Deep Residual
        Learning for Image
    Recognition (2015)
    '''

    F11, F3, F12 = filters
    initializer = tf.keras.initializers.HeNormal(seed=None)
    # Save the input value. You'll need this later to
    # add back to the main path.
    X_shortcut = A_prev
    # First component of main path
    X = tf.keras.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=1,
        padding='valid',
        kernel_initializer=initializer
    )(A_prev)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    # Second component of main path
    X = tf.keras.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    # Third component of main path
    X = tf.keras.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=1,
        padding='valid',
        kernel_initializer=initializer
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    # Final step: Add shortcut value to main path, and pass it through a RELU
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    return X
