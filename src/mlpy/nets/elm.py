# SYSTEM IMPORTS
import numpy
import os
import random
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
import basenet


class ELM(basenet.BaseNet):
    def __init__(self, layer_sizes, seed=None, activation_func_ptr=sigmoid):
        super(ELM, self).__init__(layer_sizes, seed=seed, activation_func_ptr=activation_func_ptr)

        assert(self.num_layers == 3)

        self.biases = self.biases[:-1]

    def _train(self, X, Y):

        # also called the 'hidden layer output matrix' computed as:
        # H is a n x m matrix (n training examples, m hidden nodes)
        # and H = [g(w_1 dot x_1 + b_1), ..., g(w_m dot x_1 + b_m)]
        #         [g(w_1 dot x_2 + b_1), ..., g(w_m dot x_2 + b_m)]
        #         [       ...            ...             ...      ]
        #         [g(w_1 dot x_n + b_1), ..., g(w_m dot x_n + b_m)]
        H = self.activation_func_ptr(numpy.dot(X, self.weights[0]) + self.biases[0])

        h_pseudoinverse = numpy.linalg.pinv(H)
        self.weights[-1] = numpy.dot(h_pseudoinverse, Y)

    """ """
    def predict(self, X):
        # a = feature_vectors
        # a = self._active_func_ptr(numpy.dot(a, self._weights[0]) + self._hidden_bias)
        # a = numpy.dot(a, self._weights[-1])
        return numpy.dot(self.activation_func_ptr(numpy.dot(X, self.weights[0]) +
            self.biases[0]), self.weights[-1])
    """ """

