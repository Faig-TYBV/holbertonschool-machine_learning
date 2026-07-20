#!/usr/bin/env python3
"""Multi Head Attention module for machine translation.
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention class to perform multi head attention.
    """

    def __init__(self, dm, h):
        """Initializes the MultiHeadAttention instance.

        Args:
            dm (int): The dimensionality of the model.
            h (int): The number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Splits the last dimension of a tensor into (h, depth).

        Args:
            x (tf.Tensor): The tensor to be split.
            batch_size (int): The batch size.

        Returns:
            tf.Tensor: Transposed tensor of shape
                (batch_size, h, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Executes the forward pass for multi head attention.

        Args:
            Q (tf.Tensor): Tensor of shape (batch, seq_len_q, dk) containing
                the input to generate the query matrix.
            K (tf.Tensor): Tensor of shape (batch, seq_len_v, dk) containing
                the input to generate the key matrix.
            V (tf.Tensor): Tensor of shape (batch, seq_len_v, dv) containing
                the input to generate the value matrix.
            mask (tf.Tensor): Contains the optional mask, always None.

        Returns:
            tuple: (output, weights)
                output: A tensor with its last two dimensions as
                    (..., seq_len_q, dm) containing the scaled dot product
                    attention.
                weights: A tensor with its last three dimensions as
                    (..., h, seq_len_q, seq_len_v) containing the attention
                    weights.
        """
        batch_size = tf.shape(Q)[0]

        # Pass inputs through the dense layers
        q = self.Wq(Q)  # (batch_size, seq_len_q, dm)
        k = self.Wk(K)  # (batch_size, seq_len_v, dm)
        v = self.Wv(V)  # (batch_size, seq_len_v, dm)

        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Calculate scaled dot product attention
        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)

        # Transpose and reshape scaled attention
        # (batch_size, seq_len_q, h, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, dm)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        # Pass through the linear layer
        output = self.linear(concat_attention)  # (batch_size, seq_len_q, dm)

        return output, attention_weights
