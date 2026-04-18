#!/usr/bin/env python3
'''4-resnet50.py'''


from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015).

    Returns:
        Keras model
    """

    initializer = K.initializers.HeNormal(seed=0)
    inputs = K.Input(shape=(224, 224, 3))
    # ── Stage 1 ──────────────────────────────────────────────────────────────
    # Conv1: 7x7, 64 filters, stride 2 → 112x112x64
    X = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                        padding='same',
                        kernel_initializer=initializer)(inputs)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    # MaxPool: 3x3, stride 2 → 56x56x64
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='same')(X)
    # ── Stage 2 (Conv2_x) ────────────────────────────────────────────────────
    # Projection block (s=1, no spatial downsampling, but channel expansion)
    X = projection_block(X, filters=(64, 64, 256), s=1)
    X = identity_block(X, filters=(64, 64, 256))
    X = identity_block(X, filters=(64, 64, 256))
    # ── Stage 3 (Conv3_x) ────────────────────────────────────────────────────
    # Projection block (s=2, spatial downsampling 56→28)
    X = projection_block(X, filters=(128, 128, 512), s=2)
    X = identity_block(X, filters=(128, 128, 512))
    X = identity_block(X, filters=(128, 128, 512))
    X = identity_block(X, filters=(128, 128, 512))
    # ── Stage 4 (Conv4_x) ────────────────────────────────────────────────────
    # Projection block (s=2, spatial downsampling 28→14)
    X = projection_block(X, filters=(256, 256, 1024), s=2)
    X = identity_block(X, filters=(256, 256, 1024))
    X = identity_block(X, filters=(256, 256, 1024))
    X = identity_block(X, filters=(256, 256, 1024))
    X = identity_block(X, filters=(256, 256, 1024))
    X = identity_block(X, filters=(256, 256, 1024))
    # ── Stage 5 (Conv5_x) ────────────────────────────────────────────────────
    # Projection block (s=2, spatial downsampling 14→7)
    X = projection_block(X, filters=(512, 512, 2048), s=2)
    X = identity_block(X, filters=(512, 512, 2048))
    X = identity_block(X, filters=(512, 512, 2048))
    # ── Classifier head ──────────────────────────────────────────────────────
    # Average pool: 7x7 → 1x1x2048
    X = K.layers.AveragePooling2D(pool_size=(7, 7), padding='same')(X)
    # Flatten + Dense softmax → 1000 classes
    X = K.layers.Flatten()(X)
    X = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=initializer)(X)
    return K.Model(inputs=inputs, outputs=X)
