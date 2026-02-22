#!/usr/bin/env python3
"""Specificity Calculation"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class."""

    # Total number of samples
    total = np.sum(confusion)

    # True positives for each class
    true_positives = np.diag(confusion)

    # False positives for each class (column sum minus TP)
    false_positives = np.sum(confusion, axis=0) - true_positives

    # False negatives for each class (row sum minus TP)
    false_negatives = np.sum(confusion, axis=1) - true_positives

    # True negatives for each class
    true_negatives = total - (true_positives)
    true_negatives -= (false_positives + false_negatives)

    # Specificity calculation
    return true_negatives / (true_negatives + false_positives)
