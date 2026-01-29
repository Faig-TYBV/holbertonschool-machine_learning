#!/usr/bin/env python3
'''elementwise operations in numpy arrays'''


def np_elementwise(mat1, mat2):
    '''elementwise operations of mat2 and mat1'''

    return tuple((mat1+mat2, mat1-mat2, mat1*mat2, mat1/mat2))
