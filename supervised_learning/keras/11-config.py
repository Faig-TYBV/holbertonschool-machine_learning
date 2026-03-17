#!/usr/bin/env python3
'''11-config.py
    - Save and load model configuration with Keras
'''


import tensorflow.keras as K


def save_config(network, filename):
    '''save_config - saves a model's configuration in JSON format'''

    json_config = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_config)
    return None


def load_config(filename):
    '''load_config - loads a model with a specific configuration from JSON'''

    with open(filename, 'r') as f:
        json_config = f.read()
    # Reconstruct the model architecture from the JSON string
    return K.models.model_from_json(json_config)
