#!/usr/bin/env python3
"""
Contains the Binomial class for probability distribution calculations
"""


class Binomial:
    """
    Represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(round(n))
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            sum_diff_sq = sum([(x - mean) ** 2 for x in data])
            variance = sum_diff_sq / len(data)

            p_initial = 1 - (variance / mean)
            self.n = int(round(mean / p_initial))
            self.p = float(mean / self.n)

            if self.p <= 0 or self.p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes"
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        def factorial(num):
            """Helper to calculate factorial"""
            res = 1
            for i in range(1, num + 1):
                res *= i
            return res

        n_fact = factorial(self.n)
        k_fact = factorial(k)
        nmk_fact = factorial(self.n - k)

        n_choose_k = n_fact / (k_fact * nmk_fact)
        return n_choose_k * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of "successes"
        """
        k = int(k)
        if k < 0:
            return 0
        if k >= self.n:
            return 1

        probability = 0
        for i in range(k + 1):
            probability += self.pmf(i)
        return float(probability)
