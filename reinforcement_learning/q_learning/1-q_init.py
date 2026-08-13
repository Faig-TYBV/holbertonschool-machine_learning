#!/usr/bin/env python3
"""
Module containing a function to initialize a Q-table for
reinforcement learning.
"""
import numpy as np


def q_init(env):
    """
    Initializes the Q-table for a given environment.

    Parameters:
    - env: The FrozenLakeEnv instance from gymnasium.

    Returns:
    - The Q-table as a numpy.ndarray of zeros, with dimensions corresponding
      to the number of states and actions in the environment.
    """
    # Retrieve the number of states and actions from the environment
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize the Q-table with zeros
    q_table = np.zeros((num_states, num_actions))

    return q_table
