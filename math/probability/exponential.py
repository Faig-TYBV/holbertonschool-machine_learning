#!/usr/bin/env python3
"""
Contains the Exponential class for probability distribution calculations
"""


class Exponential:
    """
    Represents an exponential distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Exponential distribution
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # lambtha is the inverse of the mean
            self.lambtha = float(sum(data) / len(data)) ** -1

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period x
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return float(self.lambtha * e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period x
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return float(1 - e ** (-self.lambtha * x))