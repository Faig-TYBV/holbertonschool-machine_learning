#!/usr/bin/env python3
'''finding sum of 2D arrays'''


def add_matrices2D(mat1, mat2):
    '''finding sum of 2D arrays'''

    if len(mat2) != len(mat1):
        return None
    arr3 = []
    for i in range(len(mat1)):
        row = []
        if len(mat1[i]) != len(mat2[i]):
            return None
        for j in range(len(mat1[i])):
            row.append(mat1[i][j]+mat2[i][j])
        arr3.append(row)
    return arr3
