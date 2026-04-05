#!/usr/bin/env python3
'''2-conv_backward.py'''


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    '''performs back propagation over a convolutional layer of a neural network
    '''

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    _, h_new, w_new, _ = dZ.shape
    sh, sw = stride
    if padding == "same":
        ph = max((h_prev - 1) * sh + kh - h_prev, 0)
        pw = max((w_prev - 1) * sw + kw - w_prev, 0)
        pad_top,  pad_bottom = ph // 2, ph - ph // 2
        pad_left, pad_right = pw // 2, pw - pw // 2
        A_prev_pad = np.pad(
            A_prev,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant"
        )
    else:
        pad_top = pad_bottom = pad_left = pad_right = 0
        A_prev_pad = A_prev
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)   # (1, 1, 1, c_new)
    for i in range(h_new):
        for j in range(w_new):
            hs, ws = i * sh, j * sw
            # dZ slice:  (m, c_new)
            dz = dZ[:, i, j, :]
            # A_prev_pad slice: (m, kh, kw, c_prev)
            a_slice = A_prev_pad[:, hs:hs + kh, ws:ws + kw, :]
            # dW  += A_slice^T · dz  →  (kh, kw, c_prev, c_new)
            dW += np.tensordot(a_slice, dz, axes=([0], [0]))
            # dA_prev_pad slice += dz · W^T  →  (m, kh, kw, c_prev)
            dA_prev_pad[:, hs:hs + kh, ws:ws + kw, :] += (
                np.tensordot(dz, W, axes=([1], [3]))
            )
    # Strip padding to recover dA_prev
    if pad_bottom == 0 and pad_right == 0:
        dA_prev = dA_prev_pad[:, pad_top:, pad_left:, :]
    else:
        dA_prev = dA_prev_pad[:,
                              pad_top: -pad_bottom,
                              pad_left: -pad_right, :]
    return dA_prev, dW, db
