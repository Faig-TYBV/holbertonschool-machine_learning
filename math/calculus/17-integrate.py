#!/usr/bin/env python3
"""Finding integral"""


def poly_integral(poly, C=0):
    """Finding integral"""

    if not isinstance(poly, list) or not isinstance(C, int) or len(poly) == 0:
        return None
    integral = []
    integral.append(C)
    i = 1
    for ele in poly:
        if ele % i == 0:
            integral.append(ele//i)
        else:
            integral.append(ele/i)
        i += 1
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral
