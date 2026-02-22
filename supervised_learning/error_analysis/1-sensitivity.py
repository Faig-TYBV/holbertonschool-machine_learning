#!/usr/bin/env python3
"""
Sensitivity (Recall) Calculation

This module contains a function to compute the sensitivity for each class
from a confusion matrix.
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall) for each class.

    Sensitivity measures the proportion of actual positives that are
    correctly identified. For each class, it is defined as:

        sensitivity = TP / (TP + FN)

    where:
        - TP (True Positives) is the value on the diagonal of the confusion matrix
        - FN (False Negatives) is the sum of the remaining values in the same row

    Parameters
    ----------
    confusion : numpy.ndarray
        A confusion matrix of shape (classes, classes), where rows represent
        the true labels and columns represent the predicted labels.

    Returns
    -------
    numpy.ndarray
        A 1D array of shape (classes,) containing the sensitivity for
        each class.
    """
    # True positives are the diagonal elements
    true_positives = np.diag(confusion)

    # Total actual positives per class (row-wise sum)
    actual_positives = np.sum(confusion, axis=1)

    # Sensitivity calculation
    return true_positives / actual_positives
