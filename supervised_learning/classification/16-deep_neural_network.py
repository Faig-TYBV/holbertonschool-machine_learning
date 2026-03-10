#!/usr/bin/env python3
"""Deep Neural Network class"""


import numpy as np


class DeepNeuralNetwork:
    '''Deep Neural Network class'''

    def __init__(self, nx, layers):
        '''Constructor method for DeepNeuralNetwork class'''

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if min(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        '''He et al. initialization'''
        for i in range(self.L):
            if i == 0:
                vector = np.random.normal(0, 1, (layers[i], nx))
                self.weights['W' + str(i + 1)] = vector
                self.weights['W' + str(i + 1)] *= np.sqrt(2 / nx)
            else:
                vector = np.random.normal(0, 1, (layers[i], layers[i - 1]))
                self.weights['W' + str(i + 1)] = vector
                self.weights['W' + str(i + 1)] *= np.sqrt(2 / layers[i - 1])
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
