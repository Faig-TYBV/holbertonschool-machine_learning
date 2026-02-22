#!/usr/bin/env python3
"""Error analysis"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels and predictions.

    Parameters:
    labels (np.ndarray): shape (m, classes), one-hot true labels
    logits (np.ndarray): shape (m, classes), one-hot predicted labels

    Returns:
    np.ndarray: shape (classes, classes), confusion matrix
    """

    # Convert one-hot vectors to class indices
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    classes = labels.shape[1]

    # Initialize confusion matrix
    confusion = np.zeros((classes, classes))

    # Populate confusion matrix
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1

    return confusion
