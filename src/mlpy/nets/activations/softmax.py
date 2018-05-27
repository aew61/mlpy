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
    # Y = softmax(X)  # this is 2d, so the jacobian is shape (X.shape[0], X.shape[0])
    # if len(X.shape) == 1:
    #     Y_ = Y.reshape(-1, 1)
    #     out = numpy.sum(numpy.diagflat(Y_) - numpy.dot(Y_, Y_.T), axis=0)
    #     print(out)
    #     return out
        # return numpy.diagflat(Y_) - numpy.dot(Y_, Y_.T)

    # gradient = numpy.zeros(X.shape)
    # for r in range(X.shape[0]):
    #     y_ = Y[r].reshape(-1, 1)
    #     gradient[r] = numpy.sum(numpy.diagflat(y_) - numpy.dot(y_, y_.T), axis=0)

    # print(gradient)
    # return gradient
    return numpy.ones(X.shape)


def softmax_single_jacobian(Y):
    Y_ = Y.reshape(-1, 1)
    return numpy.diagflat(Y_) - numpy.dot(Y_, Y_.T)


def softmax_jacobian(X):
    Y = softmax(X)
    if len(Y.shape) == 1:
        return softmax_single_jacobian(Y)

    out = numpy.zeros((X.shape[0], X.shape[1], X.shape[1]), dtype=float)
    for i in range(X.shape[0]):
        out[i] = softmax_single_jacobian(Y[i])
    return out    

