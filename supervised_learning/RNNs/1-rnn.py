#!/usr/bin/env python3
"""
Module that contains the rnn function for a simple Recurrent Neural Network.
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN across all time steps.

    Args:
        rnn_cell: An instance of RNNCell
        X (numpy.ndarray): The data input, shape (t, m, i)
        h_0 (numpy.ndarray): The initial hidden state, shape (m, h)

    Returns:
        tuple: (H, Y)
            - H is a numpy.ndarray containing all of the hidden states
            - Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape

    # Initialize lists to store the hidden states and outputs
    H = [h_0]
    Y = []

    # Set the first previous hidden state to h_0
    h_prev = h_0

    # Loop through each time step
    for step in range(t):
        # Perform forward propagation for the current time step
        h_next, y_step = rnn_cell.forward(h_prev, X[step])

        # Store the computed hidden state and output
        H.append(h_next)
        Y.append(y_step)

        # Update h_prev for the next iteration
        h_prev = h_next

    # Convert the lists to numpy arrays
    H = np.array(H)
    Y = np.array(Y)

    return H, Y
