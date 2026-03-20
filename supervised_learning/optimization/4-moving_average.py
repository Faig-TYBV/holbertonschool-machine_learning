#!/usr/bin/env python3
"""4. Moving Average
"""


import numpy as np


def moving_average(data, beta):
    """Calculates the moving average of a data set."""

    moving_averages = []
    v = 0
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        moving_averages.append(v / (1 - beta ** (i + 1)))
    return moving_averages
