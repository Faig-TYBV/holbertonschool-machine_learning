#!/usr/bin/env python3
"""Module for performing K-means clustering using sklearn."""

import sklearn.cluster


def kmeans(X, k):
    """Perform K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.

    Returns:
        tuple: (C, clss) where C is a numpy.ndarray of shape (k, d)
               containing the centroid means and clss is a numpy.ndarray
               of shape (n,) containing the cluster index for each data point.
    """
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)

    C = model.cluster_centers_
    clss = model.labels_

    return C, clss
