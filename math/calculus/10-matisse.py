#!/usr/bin/env python3
"""Finding derivatives"""


def poly_derivative(poly):
    """Coefficients of the derivative of the polynomial"""

    dev = []
    if not isinstance(poly, list):
        dev.append(None)
        return None
    i = 0
    for ele in poly:
        if i == 0:
            i += 1
            continue
        dev.append(ele*i)
        i += 1
    if len(dev) == 0:
        dev.append(0)
    return dev
