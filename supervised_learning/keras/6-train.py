#!/usr/bin/env python3
'''6-train.py
    - Train a model with Keras
'''


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    '''train_model - trains a model using mini-batch gradient descent'''

    callbacks = []
    # Check if early_stopping is requested
    if early_stopping and validation_data:
        # Create the callback object
        early_stop_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop_callback)
    # Pass the callbacks list to fit()
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )
