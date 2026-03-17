#!/usr/bin/env python3
'''7-train.py
    - Train a model with Keras
'''


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    '''train_model - trains a model with learning rate decay'''

    callbacks = []

    # 1. Setup Early Stopping
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        ))

    # 2. Setup Learning Rate Decay (Stepwise Inverse Time Decay)
    if learning_rate_decay and validation_data:
        def lr_schedule(epoch):
            """Calculates the learning rate based on inverse time decay"""
            return alpha / (1 + decay_rate * epoch)

        # verbose=1 in the scheduler ensures Keras prints the update message
        callbacks.append(
            K.callbacks.LearningRateScheduler(lr_schedule, verbose=1))
    # 3. Train the model
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
