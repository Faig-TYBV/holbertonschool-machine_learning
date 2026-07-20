#!/usr/bin/env python3
"""Module that defines the DecoderBlock class for a Transformer decoder.
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Transformer Decoder Block class.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the decoder block.

        Args:
            dm (int): The dimensionality of the model.
            h (int): The number of heads.
            hidden (int): The number of hidden units in the fully
                connected layer.
            drop_rate (float): The dropout rate.
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Executes the forward pass for the decoder block.

        Args:
            x (tf.Tensor): Tensor of shape (batch, target_seq_len, dm)
                containing the input to the decoder block.
            encoder_output (tf.Tensor): Tensor of shape
                (batch, input_seq_len, dm) containing the output of the encoder.
            training (bool): A boolean to determine if the model is training.
            look_ahead_mask (tf.Tensor): The mask to be applied to the
                first multi head attention layer.
            padding_mask (tf.Tensor): The mask to be applied to the
                second multi head attention layer.

        Returns:
            tf.Tensor: A tensor of shape (batch, target_seq_len, dm)
                containing the block's output.
        """
        # Block 1: Masked Multi-head Self Attention over target sequence
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Block 2: Multi-head Cross Attention over encoder outputs
        # Query comes from out1, Key and Value come from encoder_output
        attn2, _ = self.mha2(out1, encoder_output, encoder_output,
                             padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # Block 3: Point-wise Feed Forward Network
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3
