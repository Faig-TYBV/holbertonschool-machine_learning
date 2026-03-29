#!/usr/bin/env python3
"""0-convolve_grayscale_valid.py
"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """performs a valid convolution on grayscale images with custom padding
    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for
                the convolution"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    convolved_h = h - kh + 1
    convolved_w = w - kw + 1
    convolved_images = np.zeros((m, convolved_h, convolved_w))
    for i in range(convolved_h):
        for j in range(convolved_w):
            region = images[:, i:i + kh, j:j + kw]
            convolved_images[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    return convolved_images
