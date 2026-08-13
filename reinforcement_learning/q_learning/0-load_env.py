#!/usr/bin/env python3
"""
Module containing the function to load the FrozenLake environment
from the gymnasium library.
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.

    Parameters:
    - desc (list of lists, optional): A custom description of the map to
      load for the environment.
    - map_name (str, optional): The pre-made map to load.
    - is_slippery (bool, optional): Determines if the ice is slippery.
      Defaults to False.

    Returns:
    - The gymnasium environment instance for FrozenLake.
    """
    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi"
    )
