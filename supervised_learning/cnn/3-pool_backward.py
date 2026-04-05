#!/usr/bin/env python3
'''3-pool_backward.py'''


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''performs back propagation over a pooling layer of a neural network
    '''

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    _, h_out, w_out, _ = dA.shape
    dA_prev = np.zeros_like(A_prev)
    for i in range(h_out):
        for j in range(w_out):
            hs, ws = i * sh, j * sw
            da = dA[:, i, j, :]
            if mode == 'max':
                window = A_prev[:, hs:hs + kh, ws:ws + kw, :]
                max_vals = np.max(window, axis=(1, 2), keepdims=True)
                mask = (window == max_vals)
                # Normalise mask so tied maxima share the gradient equally
                mask = mask / mask.sum(axis=(1, 2), keepdims=True)
                dA_prev[:, hs:hs + kh, ws:ws + kw, :] += (
                    mask * da[:, np.newaxis, np.newaxis, :]
                )
            else:  # avg
                dA_prev[:, hs:hs + kh, ws:ws + kw, :] += (
                    da[:, np.newaxis, np.newaxis, :] / (kh * kw)
                )
    return dA_prev
