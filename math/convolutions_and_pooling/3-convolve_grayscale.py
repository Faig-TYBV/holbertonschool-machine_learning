#!/usr/bin/env python3
"""3-convolve_grayscale.py
"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on grayscale images with custom padding and stride
    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for
                the convolution
        padding: either a tuple of (ph, pw), ‘same’, or ‘valid’
            if ‘same’, performs a same convolution
            if ‘valid’, performs a valid convolution
        stride: tuple of (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'same':
        pad_h = kh // 2
        pad_w = kw // 2
    elif padding == 'valid':
        pad_h = 0
        pad_w = 0
    else:
        pad_h, pad_w = padding
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')
    convolved_h = (h + 2 * pad_h - kh) // sh + 1
    convolved_w = (w + 2 * pad_w - kw) // sw + 1
    convolved_images = np.zeros((m,
                                 convolved_h, convolved_w))
    for i in range(convolved_h):
        for j in range(convolved_w):
            region = padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            convolved_images[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    return convolved_images
