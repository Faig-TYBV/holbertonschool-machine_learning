#!/usr/bin/env python3
'''
Docstring for holbertonschool-machine_learning.probability.normal
'''


class Normal:
  '''
  Normal class
  '''

  def __init__(self, data=None, mean=0., stddev=1.):
    '''
    Docstring for __init__ of normal class
    '''

    if data is None:
      if stddev <= 0:
        raise ValueError("stddev must be a positive value")
      self.mean = float(mean)
      self.stddev = float(stddev)
    else:
      if not isinstance(data, list):
        raise TypeError("data must be a list")
      if len(data) < 2:
        raise ValueError("data must contain multiple values")
      self.mean = float(sum(data) / len(data))
      sum_diff_sq = sum((x-self.mean)**2 for x in data)
      variance = sum_diff_sq / len(data)
      self.stddev = float(variance**0.5)

  def z_score(self, x):
    '''calculating z-score'''

    return (x - self.mean) / self.stddev
  
  def x_value(self, z):
    '''calculating x-value'''

    return z * self.stddev + self.mean
  
  def pdf(self, x):
    '''
    Calculates the value of the PDF for a given x-value
    '''

    pi = 3.1415926536
    e = 2.7182818285
    exponent = -0.5 * (self.z_score(x) ** 2)
    coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)
    return coefficient * (e ** exponent)
  def cdf(self, x):
    '''calculating cdf'''

    pi = 3.1415926536
    e = 2.7182818285
    z = self.z_score(x)
    X = z / (2 ** 0.5)
    erf = 2 / pi * (X - X * X * X /3 +  X ** 5 / 10 - X ** 7 / 42 + X ** 9 / 216)
    return float(0.5 * (1 + erf))

      

  


