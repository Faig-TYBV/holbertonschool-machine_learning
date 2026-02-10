#!/usr/bin/env python3
"""
Contains the Poisson class for probability distribution calculations
"""


class Poisson:
    """
    Represents a Poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution
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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes"
        """
        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)

        e = 2.7182818285
        result = e ** -self.lambtha
        factorial_k = 1.0
        lambtha_pow_k = 1.0

        for i in range(1, k + 1):
            factorial_k *= i
            lambtha_pow_k *= self.lambtha

        return (result * lambtha_pow_k) / factorial_k

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of "successes"
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        total_prob = 0
        for i in range(k + 1):
            total_prob += self.pmf(i)
        return total_prob
