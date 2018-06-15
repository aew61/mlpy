# SYSTEM IMPORTS
import numpy


# PYTHON PROJECT IMPORTS


def cross_entropy(P, Y):
    assert(P.shape == Y.shape)
    return numpy.sum(-numpy.log(P[numpy.arange(P.shape[0]), numpy.argmax(Y, axis=1)])) / P.shape[0]
    # return numpy.sum(-numpy.log(P[Y == 1])) / P.shape[0]


def cross_entropy_error(P, Y):
    assert(P.shape == Y.shape)
    return P - Y

