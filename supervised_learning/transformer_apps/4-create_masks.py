#!/usr/bin/env python3
"""Creates all masks for training/validation of a transformer"""
import tensorflow as tf


def create_padding_mask(seq):
    """Creates a padding mask for a batch of tokenized sequences

    Args:
        seq: tf.Tensor of shape (batch_size, seq_len) containing
            tokenized sentences, padded with 0s

    Returns:
        a tf.Tensor of shape (batch_size, 1, 1, seq_len) containing 1s
        where seq is padding and 0s elsewhere
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """Creates a look ahead mask to mask future tokens in a sequence

    Args:
        size: the length of the sequence to mask

    Returns:
        a tf.Tensor of shape (size, size) containing 1s where a token
        should be masked (future positions) and 0s elsewhere
    """
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


def create_masks(inputs, target):
    """Creates all masks for training/validation

    Args:
        inputs: a tf.Tensor of shape (batch_size, seq_len_in) that
            contains the input sentence
        target: a tf.Tensor of shape (batch_size, seq_len_out) that
            contains the target sentence

    Returns:
        encoder_mask, combined_mask, decoder_mask
            encoder_mask: the tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in) to be applied in the
                encoder
            combined_mask: the tf.Tensor of shape
                (batch_size, 1, seq_len_out, seq_len_out) used in the
                1st attention block in the decoder to pad and mask
                future tokens in the input received by the decoder
            decoder_mask: the tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in) used in the 2nd
                attention block in the decoder
    """
    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)

    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len_out)
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
