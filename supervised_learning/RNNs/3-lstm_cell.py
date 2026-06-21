#!/usr/bin/env python3
"""
Module that contains the LSTMCell class for a Long Short-Term Memory unit.
"""

import numpy as np


class LSTMCell:
    """Represents a Long Short-Term Memory (LSTM) cell."""

    def __init__(self, i, h, o):
        """
        Initializes the LSTMCell instance.

        Args:
            i (int): Dimensionality of the data
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        # Initialize weights exactly in the specified order
        # Weight matrices for the concatenated [h_prev, x_t]
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        # Weight matrix for the output
        self.Wy = np.random.normal(size=(h, o))

        # Initialize biases as zeros
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state, shape (m, h)
            c_prev (numpy.ndarray): Previous cell state, shape (m, h)
            x_t (numpy.ndarray): Data input for the cell, shape (m, i)

        Returns:
            tuple: (h_next, c_next, y)
        """
        # Concatenate previous hidden state and input data
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate (f), Update gate (u), and Output gate (o) using sigmoid
        f = 1 / (1 + np.exp(-(np.matmul(concat, self.Wf) + self.bf)))
        u = 1 / (1 + np.exp(-(np.matmul(concat, self.Wu) + self.bu)))
        o = 1 / (1 + np.exp(-(np.matmul(concat, self.Wo) + self.bo)))

        # Intermediate cell state (candidate)
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        # Next cell state
        c_next = f * c_prev + u * c_tilde

        # Next hidden state
        h_next = o * np.tanh(c_next)

        # Calculate raw output
        z = np.matmul(h_next, self.Wy) + self.by

        # Apply softmax activation (using max subtraction for stability)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, c_next, y
