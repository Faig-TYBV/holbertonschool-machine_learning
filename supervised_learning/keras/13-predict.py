#!/usr/bin/env python3
'''13-predict.py
    - Make predictions with a trained model
'''


import tensorflow.keras as K


def predict(network, data, verbose=False):
    '''predict_model - makes predictions with a trained model'''

    results = network.predict(x=data, verbose=verbose)
    return results
