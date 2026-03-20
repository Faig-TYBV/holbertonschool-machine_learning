#!/usr/bin/env python3
"""1. Normalize
"""


import numpy as np


def normalize(X, m, s):
    """Normalizes a matrix using the mean and standard deviation."""

    return (X - m) / s
