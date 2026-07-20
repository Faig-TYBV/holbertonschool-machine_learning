#!/usr/bin/env python3
"""Module to calculate positional encoding for a transformer.
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer.

    Args:
        max_seq_len (int): The maximum sequence length.
        dm (int): The model depth.

    Returns:
        numpy.ndarray: A numpy.ndarray of shape (max_seq_len, dm)
            containing the positional encoding vectors.
    """
    P = np.arange(max_seq_len)[:, np.newaxis]
    Q = np.arange(dm)[np.newaxis, :]

    # Calculate the continuous angles for all positions and dimensions
    angles = P / (10000 ** ((2 * (Q // 2)) / dm))

    # Apply sin to even indices and cos to odd indices
    PE = np.zeros((max_seq_len, dm))
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    return PE
