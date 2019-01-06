# SYSTEM IMPORTS
import numpy


# PYTHON PROJECT IMPORTS


def squared_difference(Y_hat, Y):
    assert(Y_hat.shape == Y.shape)
    return 0.5 * numpy.sum((Y_hat - Y) ** 2) / Y_hat.shape[0]


def squared_difference_error(Y_hat, Y):
    assert(Y_hat.shape == Y.shape)
    return Y_hat - Y

