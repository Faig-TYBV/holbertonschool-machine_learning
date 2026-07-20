#!/usr/bin/env python3
"""Module that defines the Encoder class for a Transformer.
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Transformer Encoder class.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initializes the Encoder.

        Args:
            N (int): The number of blocks in the encoder.
            dm (int): The dimensionality of the model.
            h (int): The number of heads.
            hidden (int): The number of hidden units in the fully
                connected layer.
            input_vocab (int): The size of the input vocabulary.
            max_seq_len (int): The maximum sequence length possible.
            drop_rate (float): The dropout rate. Defaults to 0.1.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Executes the forward pass for the encoder.

        Args:
            x (tf.Tensor): A tensor containing the input to the encoder.
                Expected shape is (batch, input_seq_len).
            training (bool): A boolean to determine if the model is training.
            mask (tf.Tensor): The mask to be applied for multi head attention.

        Returns:
            tf.Tensor: A tensor of shape (batch, input_seq_len, dm)
                containing the encoder output.
        """
        seq_len = tf.shape(x)[1]

        # Convert input word indices to embeddings
        x = self.embedding(x)  # (batch, input_seq_len, dm)

        # Slice the positional encoding for the sequence length and cast
        pos_encoding = self.positional_encoding[:seq_len, :]
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        
        # Expand dimensions to broadcast over the batch size
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)

        # Add positional encoding to the embeddings
        x += pos_encoding

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through each encoder block
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
