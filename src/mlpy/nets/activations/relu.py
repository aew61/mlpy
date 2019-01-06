# SYSTEM IMPORTS


# PYTHON PROJECT IMPORTS


def relu(X):
    if not hasattr(X, "shape"):
        if X > 0:
            return X
        return 0
    Y = numpy.array(X)
    Y[Y <= 0] = 0
    return Y 


def relu_prime(X):
    if not hasattr(X, "shape"):
        if X > 0:
            return 1
        return 0
    Y = numpy.zeros(X.shape)
    Y[Y > 0] = 1
    return Y

