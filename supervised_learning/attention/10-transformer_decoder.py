#!/usr/bin/env python3
"""Module that defines the Decoder class for a Transformer.
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Transformer Decoder class.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initializes the Decoder.

        Args:
            N (int): The number of blocks in the decoder.
            dm (int): The dimensionality of the model.
            h (int): The number of heads.
            hidden (int): The number of hidden units in the fully
                connected layer.
            target_vocab (int): The size of the target vocabulary.
            max_seq_len (int): The maximum sequence length possible.
            drop_rate (float): The dropout rate. Defaults to 0.1.
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Executes the forward pass for the decoder.

        Args:
            x (tf.Tensor): A tensor of shape (batch, target_seq_len)
                containing the input to the decoder.
            encoder_output (tf.Tensor): A tensor of shape
                (batch, input_seq_len, dm) containing the output
                of the encoder.
            training (bool): A boolean to determine if the model is training.
            look_ahead_mask (tf.Tensor): The mask to be applied to the
                first multi head attention layer.
            padding_mask (tf.Tensor): The mask to be applied to the
                second multi head attention layer.

        Returns:
            tf.Tensor: A tensor of shape (batch, target_seq_len, dm)
                containing the decoder output.
        """
        seq_len = tf.shape(x)[1]

        # Convert target word indices to embeddings
        x = self.embedding(x)  # (batch, target_seq_len, dm)

        # Scale the embeddings by multiplying by the square root of dm
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Slice the positional encoding for the sequence length and cast
        pos_encoding = self.positional_encoding[:seq_len, :]
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

        # Add positional encoding to the scaled embeddings
        x += pos_encoding

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through each decoder block
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)

        return x
