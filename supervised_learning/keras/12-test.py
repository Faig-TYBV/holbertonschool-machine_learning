#!/usr/bin/env python3
'''12-test.py
    - Test the save and load configuration functions
'''


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    '''test_model - tests the save and load configuration functions'''

    results = network.evaluate(x=data, y=labels, verbose=verbose)
    return results
