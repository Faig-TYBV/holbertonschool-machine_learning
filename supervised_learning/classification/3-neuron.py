#!/usr/bin/env python3
"""Neuron class"""


import numpy as np


class Neuron:
    """Neuron class"""

    def __init__(self, nx):
        '''Constructor method for Neuron class'''

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''W getter method'''

        return self.__W

    @property
    def b(self):
        '''b getter method'''

        return self.__b

    @property
    def A(self):
        '''A getter method'''

        return self.__A

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neuron'''

        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        ''''Calculates the cost of the model using logistic regression'''

        cost = 0
        for i in range(Y.shape[1]):
            cost += -Y[0, i] * np.log(A[0, i])
            - (1 - Y[0, i]) * np.log(1.0000001 - A[0, i])
        return cost / Y.shape[1]
