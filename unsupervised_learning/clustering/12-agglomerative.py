#!/usr/bin/env python3
"""Module for performing agglomerative clustering on a dataset."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Perform agglomerative clustering on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        dist (float): Maximum cophenetic distance for all clusters.

    Returns:
        numpy.ndarray: Cluster indices of shape (n,) for each data point.
    """
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, method='ward')

    clss = scipy.cluster.hierarchy.fcluster(
        linkage_matrix,
        t=dist,
        criterion='distance'
    )

    scipy.cluster.hierarchy.dendrogram(
        linkage_matrix,
        color_threshold=dist
    )

    plt.show()

    return clss
