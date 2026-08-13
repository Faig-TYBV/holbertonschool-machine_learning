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
    - desc (list of lists or list of str, optional): A custom description
      of the map to load for the environment.
    - map_name (str, optional): The pre-made map to load.
    - is_slippery (bool, optional): Determines if the ice is slippery.
      Defaults to False.

    Returns:
    - The gymnasium environment instance for FrozenLake.
    """
    if desc is None and map_name is None:
        # Generate a random 8x8 map if both desc and map_name are None
        desc = gym.envs.toy_text.frozen_lake.generate_random_map(size=8)
    elif desc is not None and isinstance(desc, list) and isinstance(desc[0],
                                                                    list):
        # Gymnasium expects desc to be an iterable of strings.
        # If a list of lists is provided, join the inner lists into strings.
        desc = ["".join(row) for row in desc]

    # Create and return the FrozenLake-v1 environment
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )

    return env
