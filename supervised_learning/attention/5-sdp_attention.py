#!/usr/bin/env python3
"""Scaled Dot Product Attention module.
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention.

    Args:
        Q (tf.Tensor): Tensor with its last two dimensions as
            (..., seq_len_q, dk) containing the query matrix.
        K (tf.Tensor): Tensor with its last two dimensions as
            (..., seq_len_v, dk) containing the key matrix.
        V (tf.Tensor): Tensor with its last two dimensions as
            (..., seq_len_v, dv) containing the value matrix.
        mask (tf.Tensor): Optional tensor broadcastable to
            (..., seq_len_q, seq_len_v) containing the mask. Defaults to None.

    Returns:
        tuple: (output, weights)
            output: A tensor with its last two dimensions as
                (..., seq_len_q, dv) containing the scaled dot product attention.
            weights: A tensor with its last two dimensions as
                (..., seq_len_q, seq_len_v) containing the attention weights.
    """
    # Calculate Q * K^T
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale by the square root of the depth (dk)
    dk = tf.cast(tf.shape(Q)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply the mask if it is provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax on the last axis (seq_len_v) to normalize weights
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply by the value matrix V
    output = tf.matmul(weights, V)

    return output, weights
