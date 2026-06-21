#!/usr/bin/env python3
"""
Module that contains the GRUCell class for a Gated Recurrent Unit.
"""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit (GRU) cell."""

    def __init__(self, i, h, o):
        """
        Initializes the GRUCell instance.

        Args:
            i (int): Dimensionality of the data
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        # Initialize weights exactly in the specified order
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

        # Initialize biases
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state, shape (m, h)
            x_t (numpy.ndarray): Data input for the cell, shape (m, i)

        Returns:
            tuple: (h_next, y)
        """
        # Concatenate previous hidden state and input data
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate (z) and reset gate (r) using sigmoid activation
        z = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))
        r = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))

        # Intermediate hidden state
        # Apply reset gate 'r' element-wise to previous hidden state
        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_r, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # Calculate raw output
        v = np.matmul(h_next, self.Wy) + self.by

        # Apply softmax activation (using max subtraction for stability)
        exp_v = np.exp(v - np.max(v, axis=1, keepdims=True))
        y = exp_v / np.sum(exp_v, axis=1, keepdims=True)

        return h_next, y
