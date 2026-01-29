#!/usr/bin/env python3
'''Matrix multipication'''


def mat_mul(mat1, mat2):
    '''Matrix Multipication'''

    if len(mat1[0]) != len(mat2):
        return None
    mat3 = []
    for row in mat1:
        mat = []
        for j in range(len(mat2[0])):
            sm = 0;
            for r in range(len(row)):
                sm = sm + row[r] * mat2[r][j]
            mat.append(sm)
        mat3.append(mat)
    return mat3
