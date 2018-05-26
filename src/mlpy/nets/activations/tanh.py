# SYSTEM IMPORTS
import numpy


# PYTHON PROJECT IMPORTS


def tanh(X):
    return numpy.tanh(X)

def tanh_prime(X):
    return 1 - numpy.power(numpy.tanh(X), 2)

