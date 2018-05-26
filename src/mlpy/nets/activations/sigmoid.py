# SYSTEM IMPORTS
import numpy


# PYTHON PROJECT IMPORTS


def sigmoid(X):
    return 1.0 / (1.0 + numpy.exp(-X))


def sigmoid_prime(X):
    Z = sigmoid(X)
    # assert(numpy.array_equal(Z*(1-Z), (1-Z)*Z))
    return Z*(1 - Z)

