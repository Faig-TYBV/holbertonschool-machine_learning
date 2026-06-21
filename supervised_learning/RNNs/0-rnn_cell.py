#!/usr/bin/env python3
'''This module contains the RNNCell class,
which represents a single cell of a simple RNN.'''


import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """
        Initializes the RNNCell instance.

        Args:
            i (int): Dimensionality of the data
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        # Weights for concatenated hidden state and input data
        # h_prev shape is (m, h) and x_t shape is (m, i)
        # so concat is (m, h + i)
        # Using weights on the right means Wh must be (h + i, h)
        self.Wh = np.random.normal(size=(h + i, h))

        # Weights for the output, shape must be (h, o)
        self.Wy = np.random.normal(size=(h, o))

        # Biases initialized as zeros
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
        # along the feature axis
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Calculate next hidden state (standard tanh activation for RNNs)
        h_next = np.tanh(np.matmul(concat_input, self.Wh) + self.bh)

        # Calculate raw output
        z = np.matmul(h_next, self.Wy) + self.by

        # Apply softmax activation (using max subtraction for
        # numerical stability)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, y
