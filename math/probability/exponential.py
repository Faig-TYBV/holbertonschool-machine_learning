#!/usr/bin/env python3
'''
Docstring for holbertonschool-machine_learning.probability.exponential
'''


class Exponential:
  '''
  Exponential class
  '''

  def __init__(self, data=None, lambtha=1.):
    '''
    Docstring for __init__
    calculating instance variable lambtha
    '''

    if data is None:
      if lambtha <= 0:
        raise ValueError("lambtha must be a positive value")
      self.lambtha = float(lambtha)
    else:
      if not isinstance(data, list):
        raise TypeError("data must be a list")
      if len(data) < 2:
        raise ValueError("data must contain multiple values")
      self.lambtha = float(sum(data) / len(data)) ** -1
  
  def pdf(self, x):
    '''Calculating pdf'''

    if x < 0:
      return 0
    else:
      e = 2.7182818285
      return float(self.lambtha * e ** (-self.lambtha * x))
    
  def cdf(self, x):
    '''Calculating cdf'''

    if x < 0:
      return 0
    e = 2.7182818285
    return 1 - e ** (-self.lambtha * x)

  


