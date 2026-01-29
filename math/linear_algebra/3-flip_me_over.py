#!/usr/bin/env python3
"""finding transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """finding transpose of a 2D matrix"""

    transposed = []
    k = len(matrix[0])
    for i in range(k):
        curr_col = []
        for row in matrix:
            curr_col.append(row[i])
        transposed.append(curr_col)
    return transposed
