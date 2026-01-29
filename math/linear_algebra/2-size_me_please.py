#!/usr/bin/env python3
def matrix_shape(matrix):
    '''finding dimensions of the matrix'''

    dimensions = []
    dimensions.append(len(matrix))
    k = matrix
    while isinstance(k[0], list):
        dimensions.append(len(k[0]))
        k = k[0]
    return dimensions
