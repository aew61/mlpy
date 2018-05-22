# SYSTEM IMPORTS
import numpy


# PYTHON PROJECT IMPORTS


def softmax(X):
    # this general process of Y = exp(Y - max(X)); return Y / sum(Y) is equivalent
    # to exp(X) / sum(exp(X)), but is more numerically stable.

    if len(X.shape) == 1:
        Y = numpy.exp(X - numpy.max(X))
        return Y / numpy.sum(Y)
    else:
        Y = numpy.exp(X - numpy.max(X, axis=1, keepdims=True))
        return Y / numpy.sum(Y, axis=1, keepdims=True)


def softmax_prime(X):
    return numpy.ones(X.shape)

