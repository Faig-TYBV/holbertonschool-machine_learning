#!/usr/bin/env python3
"""Two decay rates plotting module"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """Two decay rates plotting function"""

    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y1, label='C-14', color='red', linestyle='dashed')
    plt.plot(x, y2, label='Ra-226', color='green')
    plt.ylabel('Fraction Remaining')
    plt.xlabel('Time (years)')
    plt.title('Exponential Decay of Radioactive Elements')
    plt.ylim(0, 1)
    plt.xlim(0, 20000)
    plt.legend()
    plt.show()
