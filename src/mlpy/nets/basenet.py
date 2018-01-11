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
    def __init__(self, layer_sizes, seed=None, activation_func_ptr=sigmoid, ignore_biases=False):
        super(BaseNet, self).__init__()
        numpy.random.seed(seed)

        self.ignore_biases = ignore_biases

        filtered_layers = [l for l in layer_sizes if l > 0]
        self.num_layers = len(filtered_layers)
        self.layers = filtered_layers

        self.weights = [numpy.array([numpy.random.uniform(-1.0, 1.0)
                                     for z in range(row * col)]).reshape(row, col)
                        for row, col in zip(filtered_layers[:-1], filtered_layers[1:])]

        if self.ignore_biases:
            self.biases = [numpy.zeros(tuple([1, n])) for n in filtered_layers[1:]]
        else:
            self.biases = [numpy.array([numpy.random.uniform(-1.0, 1.0)
                                        for z in range(n)]).reshape(1, n)
                           for n in filtered_layers[1:]]

        self.activation_func_ptr = activation_func_ptr

    def _traverse(self, weights, biases):
        return zip(weights, biases)

    def feed_forward(self, X):
        a = X
        for weight, bias in self._traverse(self.weights, self.biases):
            a = self.activation_func_ptr(numpy.dot(a, weight) + bias)
        return a

    def _predict_example(self, x):
        if not hasattr(x, "shape") or len(x.shape) == 1:
            return self.feed_forward(numpy.array([x]))
        else:
            return self.feed_forward(x)

    def predict(self, X):
        return self.feed_forward(X)

