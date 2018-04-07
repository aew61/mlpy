# SYSTEM IMPORTS
import numpy
import matplotlib.pyplot as plt
import os
from sklearn import datasets as dsets
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_cd_, "..", "..")
for _dir_ in [_cd_, _src_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import activation_functions as af
from nets import ANN


def plot_decision_boundary(pred_func, X, Y):
    x_min = numpy.min(X[:, 0]) - 0.5
    x_max = numpy.max(X[:, 0]) + 0.5

    y_min = numpy.min(X[:, 1]) - 0.5
    y_max = numpy.max(X[:, 1]) + 0.5

    h = 0.01

    XX, YY = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))

    Z = pred_func(numpy.c_[XX.ravel(), YY.ravel()])
    # print(Z.shape)
    # print(numpy.max(Z), numpy.min(Z))
    Z = Z.reshape(XX.shape)

    plt.contourf(XX, YY, Z, cmap=plt.cm.Spectral)
    # print(Y.reshape(-1).shape)
    plt.scatter(X[:, 0], X[:, 1], c=Y.reshape(-1), cmap=plt.cm.Spectral)


def plot_nets(hidden_layers, nets, X, Y):
    plt.figure()
    for i, (h, n) in enumerate(zip(hidden_layers, nets)):
        plt.subplot((len(hidden_layers) + 1) / 2, 2, i+1)
        plt.title("%s hidden layer nodes" % h)
        plot_decision_boundary(lambda X: numpy.argmax(n.predict(X), axis=1), X, Y)
    plt.show()


def train(layers, learning_rate, weight_decay_coeff, training_iter, X, Y):
    Y_ = numpy.zeros((Y.shape[0], layers[-1]))
    Y_[range(Y.shape[0]), Y] = 1
    n = ANN(layers, learning_rate=learning_rate, weight_decay_coeff=weight_decay_coeff,
            afuncs=[af.tanh, af.softmax],
            afunc_primes=[af.tanh_prime, af.softmax_prime]
           )
    for i in range(training_iter):
        n.train(X, Y_)
    return n


def main():
    # hyperparameters
    num_inputs = 2
    num_outputs = 2
    learning_rate = 0.01
    weight_decay_coeff=0.01
    training_iter = 20000

    X, Y = dsets.make_moons(200, noise=0.2)

    hidden_layers = [1, 2, 3, 4, 5, 20, 30, 50]  # [1, 2, 3, 4, 5, 20, 30, 50]

    ns = list()
    for h in hidden_layers:
        print("training net with [%s] hidden layer nodes" % h)
        ns.append(train([num_inputs, h, num_outputs], learning_rate, weight_decay_coeff, training_iter, X, Y))

    plot_nets(hidden_layers, ns, X, Y)

if __name__ == "__main__":
    main()

