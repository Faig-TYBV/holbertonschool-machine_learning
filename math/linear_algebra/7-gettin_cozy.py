#!/usr/bin/env python3
"""concatanating matrices based on axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatanating matrices based on axis"""

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        mat3 = [row[:] for row in mat1]
        for row in mat2:
            mat3.append(row)
        return mat3
    else:
        if len(mat1) != len(mat2):
            return None
        mat3 = []
        for i in range(len(mat1)):
            mat3.append(mat1[i]+mat2[i])
        return mat3
