#!/usr/bin/env python3
"""Self Attention module for machine translation.
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """SelfAttention class to calculate attention for machine translation.
    """

    def __init__(self, units):
        """Initializes the SelfAttention instance.

        Args:
            units (int): The number of hidden units in the alignment model.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Executes the forward pass for the attention layer.

        Args:
            s_prev (tf.Tensor): A tensor of shape (batch, units) containing
                the previous decoder hidden state.
            hidden_states (tf.Tensor): A tensor of shape
                (batch, input_seq_len, units) containing the outputs of
                the encoder.

        Returns:
            tuple: (context, weights)
                context: A tensor of shape (batch, units) that contains the
                    context vector for the decoder.
                weights: A tensor of shape (batch, input_seq_len, 1) that
                    contains the attention weights.
        """
        # Expand s_prev to shape (batch, 1, units) to broadcast
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # score shape: (batch, input_seq_len, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev_expanded) +
                                  self.U(hidden_states)))

        # weights shape: (batch, input_seq_len, 1)
        weights = tf.nn.softmax(score, axis=1)

        # context shape before reduce_sum: (batch, input_seq_len, units)
        context = weights * hidden_states

        # context shape after reduce_sum: (batch, units)
        context = tf.reduce_sum(context, axis=1)

        return context, weights
