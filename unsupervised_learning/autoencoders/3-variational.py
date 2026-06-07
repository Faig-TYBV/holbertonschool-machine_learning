#!/usr/bin/env python3
"""Module that builds a variational autoencoder."""

import tensorflow.keras as keras
from tensorflow.keras import backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder.

    Args:
        input_dims (int): Dimension of the model input.
        hidden_layers (list): Number of nodes in each encoder layer.
        latent_dims (int): Dimension of the latent space.

    Returns:
        tuple: encoder, decoder, auto.
    """
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(
            nodes,
            activation='relu'
        )(encoded)

    mean = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    log_var = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    def sampling(args):
        """Sample from the latent distribution."""
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    latent = keras.layers.Lambda(sampling)([mean, log_var])
    encoder = keras.Model(encoder_input, [latent, mean, log_var])

    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(
            nodes,
            activation='relu'
        )(decoded)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(decoded)

    decoder = keras.Model(decoder_input, decoder_output)

    auto_output = decoder(latent)
    auto = keras.Model(encoder_input, auto_output)

    def vae_loss(y_true, y_pred):
        """Calculate VAE loss."""
        rec_loss = K.binary_crossentropy(y_true, y_pred)
        rec_loss = K.sum(rec_loss, axis=-1)

        kl_loss = 1 + log_var - K.square(mean) - K.exp(log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return rec_loss + kl_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
