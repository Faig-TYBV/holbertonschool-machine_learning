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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        '''He et al. initialization'''
        for i in range(self.L):
            if i == 0:
                vector = np.random.normal(0, 1, (layers[i], nx))
                self.__weights['W' + str(i + 1)] = vector
                self.__weights['W' + str(i + 1)] *= np.sqrt(2 / nx)
            else:
                vector = np.random.normal(0, 1, (layers[i], layers[i - 1]))
                self.__weights['W' + str(i + 1)] = vector
                self.__weights['W' + str(i + 1)] *= np.sqrt(2 / layers[i - 1])
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        '''L getter method'''

        return self.__L

    @property
    def cache(self):
        '''cache getter method'''

        return self.__cache

    @property
    def weights(self):
        '''weights getter method'''

        return self.__weights

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neural network'''

        self.__cache['A0'] = X
        for i in range(self.L):
            W = self.__weights['W' + str(i + 1)]
            b = self.__weights['b' + str(i + 1)]
            A_prev = self.__cache['A' + str(i)]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(i + 1)] = A
        return A, self.__cache

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression'''

        m = Y.shape[1]
        cost = - (1 / m)
        cost *= np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
