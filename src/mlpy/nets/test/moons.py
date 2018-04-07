# SYSTEM IMPORTS
import numpy
import matplotlib.pyplot as plt
import os
import sys
from sklearn import datasets


_cd_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_cd_, "..", "..")
if _src_dir_ not in sys.path:
    sys.path.append(_src_dir_)
del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import activation_functions as af

class ANN(object):
    def __init__(self, layers, afuncs, afunc_primes):
        numpy.random.seed(0)
        self.layers = layers
        self.afuncs = afuncs
        self.afunc_primes = afunc_primes
        self.lr = 0.01

        self.weights = [numpy.array([numpy.random.uniform(-1.0, 1.0)
                                     for z in range(row * col)]).reshape(row, col)
                        for row, col in zip(layers[:-1], layers[1:])]

        self.biases = [numpy.array([numpy.random.uniform(-1.0, 1.0)
                                    for z in range(n)]).reshape(1, n)
                       for n in layers[1:]]

    def predict(self, X):
        a = X
        for afunc, w, b in zip(self.afuncs, self.weights, self.biases):
            a = afunc(numpy.dot(a, w) + b)
        return numpy.argmax(a, axis=1)

    def train(self, X, Y):
        Y_ = numpy.zeros((Y.shape[0], self.weights[-1].shape[-1]))
        Y_[range(Y.shape[0]),Y] = 1

        zs = []
        as_ = [X]
        a = X
        for afunc, w, b in zip(self.afuncs, self.weights, self.biases):
            z = numpy.dot(a, w) + b
            zs.append(z)
            a = afunc(z)
            as_.append(a)

        delta3 = as_[-1]
        # print(numpy.unique(Y))
        # print(delta3.shape)
        # print(X.shape[0])
        # print()
        # print(delta3[range(X.shape[0]), Y].shape)
        # print(delta3[:, Y].shape)
        # print(delta3[:].shape)

        # delta3[range(X.shape[0]), Y] -= 1
        delta3 -= Y_
        dW2 = (as_[-2].T).dot(delta3)
        db2 = numpy.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(self.weights[-1].T) * (1 - numpy.power(as_[-2], 2))
        dW1 = numpy.dot(X.T, delta2)
        db1 = numpy.sum(delta2, axis=0)

        dW2 += 0.01*self.weights[-1]
        dW1 += 0.01*self.weights[-2]

        self.weights[-1] -= self.lr*dW2
        self.biases[-1] -= self.lr*db2
        self.weights[-2] -= self.lr*dW1
        self.biases[-2] -= self.lr*db1




def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def main():
    X, Y = datasets.make_moons(200, noise=0.20)
    n = ANN([2, 3, 2], [numpy.tanh, af.softmax], [af.tanh_prime, lambda X: numpy.ones(X.shape)])
    Y_ = numpy.zeros((Y.shape[0], 1))
    Y_[Y] = 1
    for i in range(20000):
        n.train(X, Y)
    plot_decision_boundary(lambda X: n.predict(X), X, Y)


if __name__ == "__main__":
    main()

