import numpy as np


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    if x > 0:
        return 1
    return 0
