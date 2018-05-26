# SYSTEM IMPORTS
import numpy
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_cd_, "..")
for _dir_ in [_cd_, _src_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
from activations import sigmoid, sigmoid_prime, softmax, softmax_prime
import core
from losses import cross_entropy, squared_difference


class BaseNet(core.Base):
    def __init__(self, layers, seed=None, afuncs=None, afunc_primes=None, ignore_overflow=False, loss_func=None):
        super(BaseNet, self).__init__()
        numpy.random.seed(seed)

        self.ignore_overflow = ignore_overflow
        self.num_layers = len(layers)
        self.weights = [numpy.random.uniform(-1.0, 1.0, size=(row, col,))
                        for row, col in zip(layers[:-1], layers[1:])]

        self.biases = [numpy.random.uniform(-1.0, 1.0, size=(1, n,))
                       for n in layers[1:]]

        if afuncs is None:
            self.afuncs = [sigmoid for _ in range(len(self.weights))]
            self.afunc_primes = [sigmoid_prime for _ in range(len(self.weights))]
        else:
            self.afuncs = list(afuncs)
            self.afunc_primes = list(afunc_primes)

        self.loss_func = loss_func
        if self.loss_func is None:
            if self.afuncs[-1] == softmax and self.afunc_primes[-1] == softmax_prime:
                self.loss_func = cross_entropy
            else:
                self.loss_func = squared_difference

    def change_settings(self, new_settings):
        return numpy.seterr(**new_settings)

    def loss_function(self, X, Y):
        return self.loss_func(self.feed_forward(X), Y)

    def feed_forward(self, X):
        new_settings = dict({"over": "ignore"})
        if not self.ignore_overflow:
            new_settings["over"] = "raise"
        old_settings = self.change_settings(new_settings)

        try:
            a = X
            for afunc, weight, bias in zip(self.afuncs, self.weights, self.biases):
                a = afunc(numpy.dot(a, weight) + bias)
            return a
        except FloatingPointError:
            raise FloatingPointError("Overflow occured, please scale features")
        finally:
            self.change_settings(old_settings)

    def _predict_example(self, x):
        # if not hasattr(x, "shape") or len(x.shape) == 1:
        #     return self.feed_forward(numpy.array([x]))
        # else:
        return self.feed_forward(x)

    def predict(self, X):
        return self.feed_forward(X)

