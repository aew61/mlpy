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
from activation_functions import sigmoid
import core


class BaseNet(core.Base):
    def __init__(self, layer_sizes, seed=None, afunc_ptrs=None, ignore_biases=False, ignore_overflow=False):
        super(BaseNet, self).__init__()
        numpy.random.seed(seed)

        self.ignore_overflow = ignore_overflow
        self.ignore_biases = ignore_biases

        filtered_layers = [l for l in layer_sizes if l > 0]
        self.num_layers = len(filtered_layers)
        self.layers = filtered_layers

        self.weights = [numpy.array([numpy.random.uniform(-1.0, 1.0)
                                     for z in range(row * col)]).reshape(row, col)
                        for row, col in zip(filtered_layers[:-1], filtered_layers[1:])]

        # if self.layers[-1] == 1:
        #     self.weights[-1] = self.weights[-1].reshape(-1)

        if self.ignore_biases:
            self.biases = [numpy.zeros(tuple([1, n])) for n in filtered_layers[1:]]
        else:
            self.biases = [numpy.array([numpy.random.uniform(-1.0, 1.0)
                                        for z in range(n)]).reshape(1, n)
                           for n in filtered_layers[1:]]

        if afunc_ptrs is None:
            self.afunc_ptr = sigmoid
        else:
            self.afunc_ptr = afunc_ptrs[0]

    def _traverse(self, weights, biases):
        return zip(weights, biases)

    def feed_forward(self, X):
        old_settings = dict()
        if not self.ignore_overflow:
            old_settings = numpy.seterr(over="raise")
        else:
            old_settings = numpy.seterr(over="ignore")

        try:
            a = X
            for weight, bias in self._traverse(self.weights, self.biases):
                a = self.afunc_ptr(numpy.dot(a, weight) + bias)
            return a
        except FloatingPointError:
            raise FloatingPointError("Overflow occured, please scale features")
        finally:
            numpy.seterr(**old_settings)

    def _predict_example(self, x):
        # if not hasattr(x, "shape") or len(x.shape) == 1:
        #     return self.feed_forward(numpy.array([x]))
        # else:
        return self.feed_forward(x)

    def predict(self, X):
        return self.feed_forward(X)

