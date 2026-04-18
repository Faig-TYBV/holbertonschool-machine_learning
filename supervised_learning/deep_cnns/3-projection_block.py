#!/usr/bin/env python3
'''3-projection_block.py'''


from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning
    for Image Recognition (2015).
    Args:
        A_prev: output from the previous layer
        filters: tuple/list of (F11, F3, F12) filter counts
        s: stride of the first convolution in both main path and shortcut

    Returns:
        Activated output of the projection block
    """

    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal(seed=0)
    # --- Main path ---
    # First component: 1x1 Convolution with stride s
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(s, s),
                        padding='same',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    # Second component: 3x3 Convolution
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    # Third component: 1x1 Convolution (no activation yet)
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    # --- Shortcut path ---
    # Project A_prev to match main path dimensions (stride s, F12 filters)
    shortcut = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(s, s),
                               padding='same',
                               kernel_initializer=initializer)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)
    # --- Merge + final activation ---
    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)
    return X
