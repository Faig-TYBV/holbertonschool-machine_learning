#!/usr/bin/env python3
"""Precision Calculation"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class.
    """

    # True positives are the diagonal elements
    true_positives = np.diag(confusion)

    # Total predicted positives per class (column-wise sum)
    predicted_positives = np.sum(confusion, axis=0)

    # Precision calculation
    return true_positives / predicted_positives
