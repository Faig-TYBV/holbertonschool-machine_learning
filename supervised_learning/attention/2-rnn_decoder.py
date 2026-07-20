#!/usr/bin/env python3
"""RNN Decoder module for machine translation.
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNNDecoder class that decodes for machine translation.
    """

    def __init__(self, vocab, embedding, units, batch):
        """Initializes the RNNDecoder instance.

        Args:
            vocab (int): The size of the output vocabulary.
            embedding (int): The dimensionality of the embedding vector.
            units (int): The number of hidden units in the RNN cell.
            batch (int): The batch size.
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Executes the forward pass for the decoder.

        Args:
            x (tf.Tensor): A tensor of shape (batch, 1) containing the
                previous word in the target sequence as an index.
            s_prev (tf.Tensor): A tensor of shape (batch, units) containing
                the previous decoder hidden state.
            hidden_states (tf.Tensor): A tensor of shape
                (batch, input_seq_len, units) containing the outputs of
                the encoder.

        Returns:
            tuple: (y, s)
                y: A tensor of shape (batch, vocab) containing the output
                   word distribution in the target vocabulary.
                s: A tensor of shape (batch, units) containing the new
                   decoder hidden state.
        """
        # context shape: (batch, units)
        context, weights = self.attention(s_prev, hidden_states)

        # x shape after embedding: (batch, 1, embedding)
        x = self.embedding(x)

        # Expand context to shape (batch, 1, units) to allow concatenation
        context_expanded = tf.expand_dims(context, 1)

        # Concatenate context vector with x in that order
        # Output shape: (batch, 1, units + embedding)
        concat_input = tf.concat([context_expanded, x], axis=-1)

        # Pass through the GRU layer
        outputs, s = self.gru(concat_input)

        # Squeeze the outputs from (batch, 1, units) to (batch, units)
        outputs = tf.squeeze(outputs, axis=1)

        # Pass through the Fully Connected layer F
        y = self.F(outputs)

        return y, s
