#!/usr/bin/env python3
"""Module that defines the Transformer class for a Transformer network.
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Transformer Network class.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initializes the Transformer network.

        Args:
            N (int): The number of blocks in the encoder and decoder.
            dm (int): The dimensionality of the model.
            h (int): The number of heads.
            hidden (int): The number of hidden units in the fully
                connected layers.
            input_vocab (int): The size of the input vocabulary.
            target_vocab (int): The size of the target vocabulary.
            max_seq_input (int): The maximum sequence length possible
                for the input.
            max_seq_target (int): The maximum sequence length possible
                for the target.
            drop_rate (float): The dropout rate. Defaults to 0.1.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """Executes the forward pass for the Transformer network.

        Args:
            inputs (tf.Tensor): A tensor of shape (batch, input_seq_len)
                containing the inputs.
            target (tf.Tensor): A tensor of shape (batch, target_seq_len)
                containing the target.
            training (bool): A boolean to determine if the model is training.
            encoder_mask (tf.Tensor): The padding mask to be applied to
                the encoder.
            look_ahead_mask (tf.Tensor): The look ahead mask to be applied
                to the decoder.
            decoder_mask (tf.Tensor): The padding mask to be applied to
                the decoder.

        Returns:
            tf.Tensor: A tensor of shape (batch, target_seq_len, target_vocab)
                containing the transformer output.
        """
        # Pass inputs through the encoder
        # Output shape: (batch_size, input_seq_len, dm)
        enc_output = self.encoder(inputs, training, encoder_mask)

        # Pass target and encoder output through the decoder
        # Output shape: (batch_size, target_seq_len, dm)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)

        # Pass decoder output through the final dense layer
        # Output shape: (batch_size, target_seq_len, target_vocab)
        final_output = self.linear(dec_output)

        return final_output
