#!/usr/bin/env python3
'''concatanating numpy arrays based on axis'''
import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''concatanating mat1 and mat2 based on axis'''

    return np.concatenate((mat1, mat2), axis)
