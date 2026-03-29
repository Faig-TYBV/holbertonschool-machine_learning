#!/usr/bin/env python3
"""5-convolve.py
"""


import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """performs a convolution on images with custom padding and stride
    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
                images
        kernels: numpy.ndarray with shape (kh, kw, c, nc) containing
        the kernels for
                the convolution
        padding: a string that is either 'same' or 'valid'
        stride: a tuple of (sh, sw) containing the strides for the convolution
    Returns:
        A numpy.ndarray containing the convolved images"""

    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w, nc))
    for k in range(nc):
        for i in range(out_h):
            for j in range(out_w):
                region = padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                output[:, i, j, k] = np.sum(region * kernels[:, :, :, k],
                                            axis=(1, 2, 3))
    return output
