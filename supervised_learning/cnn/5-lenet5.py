#!/usr/bin/env python3
'''5-lenet5.py'''


from tensorflow import keras as K


def lenet5(X):
    '''builds a modified version of the LeNet-5 architecture using Keras
    '''

    init = K.initializers.HeNormal(seed=0)

    x = K.layers.Conv2D(6, (5, 5), padding='same',
                        activation='relu', kernel_initializer=init)(X)

    x = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = K.layers.Conv2D(16, (5, 5), padding='valid',
                        activation='relu', kernel_initializer=init)(x)

    x = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = K.layers.Flatten()(x)

    x = K.layers.Dense(120, activation='relu', kernel_initializer=init)(x)

    x = K.layers.Dense(84, activation='relu', kernel_initializer=init)(x)

    x = K.layers.Dense(10, activation='softmax', kernel_initializer=init)(x)
    model = K.Model(inputs=X, outputs=x)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
