#!/usr/bin/env python3
"""2-convolve_grayscale_padding.py
"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding
    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for
                the convolution
        padding: tuple of (ph, pw)"""

    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h, pad_w = padding
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')
    convolved_h = h + 2 * pad_h - kh + 1
    convolved_w = w + 2 * pad_w - kw + 1
    convolved_images = np.zeros((m,
                                 convolved_h, convolved_w))
    for i in range(convolved_h):
        for j in range(convolved_w):
            region = padded_images[:, i:i + kh, j:j + kw]
            convolved_images[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    return convolved_images
