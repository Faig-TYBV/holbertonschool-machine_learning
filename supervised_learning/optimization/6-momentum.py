#!/usr/bin/env python3
"""6. Momentum
"""


import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Creates the training operation for a gradient descent with
        momentum optimization algorithm in tensorflow."""

    v = tf.placeholder(tf.float32, name='v')
    grad = tf.placeholder(tf.float32, name='grad')
    var = tf.placeholder(tf.float32, name='var')

    momentum = beta1 * v + (1 - beta1) * grad
    train_op = var - alpha * momentum

    return train_op, v, grad, var
