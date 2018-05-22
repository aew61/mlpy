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
from activation_functions import sigmoid, sigmoid_prime
import core


class BaseNet(core.Base):
    def __init__(self, layers, seed=None, afuncs=None, afunc_primes=None, ignore_overflow=False):
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

    def change_settings(new_settings):
        return numpy.seterr(**new_settings)

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

