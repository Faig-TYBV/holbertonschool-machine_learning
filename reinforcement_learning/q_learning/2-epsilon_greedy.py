#!/usr/bin/env python3
"""
Module containing the function to select an action using the
epsilon-greedy exploration strategy.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Parameters:
    - Q (numpy.ndarray): The Q-table.
    - state (int): The current state.
    - epsilon (float): The exploration rate.

    Returns:
    - int: The index of the next action.
    """
    # Sample a probability p to determine exploration vs exploitation
    p = np.random.uniform()

    if p < epsilon:
        # Explore: pick a random action from all possible actions
        # Q.shape[1] gives the number of actions (columns in Q-table)
        action = np.random.randint(0, int(Q.shape[1]))
    else:
        # Exploit: pick the action with the highest Q-value for the current
        # state
        action = np.argmax(Q[state, :])

    return int(action)
