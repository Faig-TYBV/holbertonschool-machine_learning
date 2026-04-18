#!/usr/bin/env python3
'''2-identity_block.py'''


from tensorflow import keras as K


def identity_block(A_prev, filters):
    '''Builds an identity block as described in
    Deep Residual Learning for Image
    Recognition (2015)
    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F11, F3, F12, respectively:
            F11: number of filters in the first 1x1 convolution
            F3: number of filters in the 3x3 convolution
            F12: number of filters in the second 1x1 convolution'''

    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)
    # Save the input value. You'll need this later
    # to add back to the main path.
    X_shortcut = A_prev
    # First component of main path
    X= K.layers.Conv2D(filters=F11, kernel_size=1, strides
                        =(1, 1), padding='valid',
                        kernel_initializer=init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    # Second component of main path
    X = K.layers.Conv2D(filters=F3, kernel_size=3, strides
                        =(1, 1), padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    # Third component of main path
    X = K.layers.Conv2D(filters=F12, kernel_size=1, strides
                        =(1, 1), padding='valid',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    # Final step: Add shortcut value to main path,
    # and pass it through a RELU
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)
    return X
