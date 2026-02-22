#!/usr/bin/env python3
"""
Sensitivity (Recall) Calculation

This module contains a function to compute the sensitivity for each class
from a confusion matrix.
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall) for each class
    """

    # True positives are the diagonal elements
    true_positives = np.diag(confusion)

    # Total actual positives per class (row-wise sum)
    actual_positives = np.sum(confusion, axis=1)

    # Sensitivity calculation
    return true_positives / actual_positives
