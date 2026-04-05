#!/usr/env/bin python3
'''0-conv_forward.py'''


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    '''performs forward propagation over a convolutional layer of a neural
    '''

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    if padding == "same":
        ph = max((h_prev - 1) * sh + kh - h_prev, 0)
        pw = max((w_prev - 1) * sw + kw - w_prev, 0)
        pad_top, pad_bottom = ph // 2, ph - ph // 2
        pad_left, pad_right = pw // 2, pw - pw // 2
        A_prev_pad = np.pad(
            A_prev,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant"
        )
    else:
        A_prev_pad = A_prev
    h_out = (A_prev_pad.shape[1] - kh) // sh + 1
    w_out = (A_prev_pad.shape[2] - kw) // sw + 1
    Z = np.zeros((m, h_out, w_out, c_new))
    for i in range(h_out):
        for j in range(w_out):
            h_start, w_start = i * sh, j * sw
            slice_ = A_prev_pad[:,
                                h_start:h_start + kh, w_start:w_start + kw, :]
            # slice_: (m, kh, kw, c_prev), W: (kh, kw, c_prev, c_new)
            Z[:, i, j, :] = np.tensordot(slice_,
                                         W, axes=([1, 2, 3],
                                                  [0, 1, 2])) + b[0, 0, 0]
    return activation(Z)
