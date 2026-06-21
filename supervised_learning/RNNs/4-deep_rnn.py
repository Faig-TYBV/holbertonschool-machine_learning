#!/usr/bin/env python3
"""
Module that contains the deep_rnn function for a Deep Recurrent Neural Network.
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells (list): List of RNNCell instances of length l
        X (numpy.ndarray): Data input, shape (t, m, i)
        h_0 (numpy.ndarray): Initial hidden state, shape (l, m, h)

    Returns:
        tuple: (H, Y)
            - H is a numpy.ndarray containing all hidden states,
              shape (t + 1, l, m, h)
            - Y is a numpy.ndarray containing all outputs,
              shape (t, m, o)
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    # Initialize lists to store the hidden states and outputs
    H = [h_0]
    Y = []

    # Keep track of the previous time step's hidden states for all layers
    h_prev_t = np.copy(h_0)

    # Loop through each time step
    for step in range(t):
        x_in = X[step]
        
        # Array to hold the new hidden states for this time step
        h_next_t = np.zeros_like(h_0)

        # Loop through each layer of the deep RNN
        for layer in range(l):
            # Forward pass for the current layer
            h_next, y_step = rnn_cells[layer].forward(h_prev_t[layer], x_in)
            
            # Store the new hidden state for this layer
            h_next_t[layer] = h_next
            
            # The output hidden state becomes the input data for the next layer
            x_in = h_next

            # If it is the last layer, capture the output y
            if layer == l - 1:
                Y.append(y_step)

        # Append the states of all layers for this time step to H
        H.append(h_next_t)
        
        # Update the previous hidden states for the next time step iteration
        h_prev_t = np.copy(h_next_t)

    return np.array(H), np.array(Y)
