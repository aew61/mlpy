# SYSTEM IMPORTS
import numpy


# PYTHON PROJECT IMPORTS


def sigmoid(X):
    return 1.0 / (1.0 + numpy.exp(-X))


def sigmoid_prime(X):
    z = sigmoid(X)
    return z * (1 - z)

