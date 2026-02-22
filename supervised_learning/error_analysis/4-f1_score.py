#!/usr/bin/env python3
"""F1 Score Calculation"""
import numpy as np

# Import previously defined functions
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class.

    The F1 score is the harmonic mean of precision and sensitivity
    (recall). For each class, it is defined as:

        F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    Parameters
    ----------
    confusion : numpy.ndarray
        A confusion matrix of shape (classes, classes), where rows represent
        the true labels and columns represent the predicted labels.

    Returns
    -------
    numpy.ndarray
        A 1D array of shape (classes,) containing the F1 score for
        each class.
    """

    # Compute precision and sensitivity using previously implemented functions
    p = precision(confusion)
    s = sensitivity(confusion)

    # F1 score calculation
    return 2 * (p * s) / (p + s)
