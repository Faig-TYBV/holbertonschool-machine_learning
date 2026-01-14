#!/usr/bin/env python3
"""Finding integral"""


def poly_integral(poly, C=0):
    """Finding integral"""

    if not isinstance(poly, list) or not isinstance(C, int):
        return None
    integral = []
    integral.append(0)
    i = 1
    for ele in poly:
        if ele%i == 0:
            integral.append(ele//i)
        else:
            integral.append(ele/i)
        i += 1
    return integral
print(poly_integral([5, 3, 0, 1]))
