#!/usr/bin/env python3
'''1-pool_forward.py'''


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''performs forward propagation over a pooling layer of a neural network
    '''

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1
    A_out = np.zeros((m, h_out, w_out, c_prev))
    pool_fn = np.max if mode == 'max' else np.mean
    for i in range(h_out):
        for j in range(w_out):
            h_start, w_start = i * sh, j * sw
            slice_ = A_prev[:, h_start:h_start + kh, w_start:w_start + kw, :]
            A_out[:, i, j, :] = pool_fn(slice_, axis=(1, 2))
    return A_out
