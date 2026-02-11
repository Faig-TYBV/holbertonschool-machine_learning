#!/usr/bin/env python3
"""
finding likelihood
"""

import numpy as np


def fac(x):
    """
    calculating factorial
    """
    s = 1
    for i in range(1, x+1):
        s *= i
    return s


def combination(a, b):
    '''
    Docstring for combination
    :param a: a
    :param b: b
    calculation kombinezon
    '''
    res = fac(a)/(fac(b) * fac(a-b))
    return res


def likelihood(x, n, P):
    """
    inside the function
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        error = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(error)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    P = P.astype(float)
    if np.any((P < 0) | (P > 1)):
        raise ValueError('All values in P must be in the range [0, 1]')
    res = combination(n, x) * (P ** x) * ((1 - P) ** (n - x))
    return res


def intersection(x, n, P, Pr):
    """
    calculating the intersection
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        error = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(error)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if np.any((P < 0) | (P > 1)):
        raise ValueError('All values in P must be in the range [0, 1]')
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')
    return (combination(n, x) * (P ** x) * ((1 - P) ** (n - x))) * Pr


def marginal(x, n, P, Pr):
    '''
    calculating the marginal
    '''
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        error = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(error)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if np.any((P < 0) | (P > 1)):
        raise ValueError('All values in P must be in the range [0, 1]')
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')
    inter = (combination(n, x) * (P ** x) * ((1 - P) ** (n - x))) * Pr
    s = 0
    for i in inter:
        s += i
    return s
