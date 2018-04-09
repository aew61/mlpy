# SYSTEM IMPORTS
import numpy


# PYTHON PROJECT IMPORTS


def softmax(X):
    Y = numpy.exp(X)
    if len(Y.shape) == 1:
        return Y / numpy.sum(Y)
    return Y / numpy.sum(Y, axis=1, keepdims=True)


def softmax_prime(X):
    return numpy.ones(X.shape)

