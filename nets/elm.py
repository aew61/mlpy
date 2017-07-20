# SYSTEM IMPORTS
import numpy
import os
import random
import sys


_current_dir_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_current_dir_, "..")
_dirs_to_add_ = [_current_dir_, _src_dir_]
for _dir_ in _dirs_to_add_:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _dirs_to_add_
del _src_dir_
del _current_dir_


# PYTHON PROJECT IMPORTS
from activation_functions import sigmoid


class elm(object):
    def __init__(self, layer_sizes, active_func_ptr=None):
        random.seed(12345)
        filtered_layers = [layer for layer in layer_sizes if layer > 0]

        assert(len(filtered_layers) == 3)
        self._layers = filtered_layers
        self._num_layers = 3

        self._weights = [numpy.array([random.gauss(0.0, 1.0)
                                      for z in range(row * col)]).reshape(row, col)
                         for row, col in zip(filtered_layers[:-1], filtered_layers[1:])]

        self._hidden_bias = numpy.array([random.gauss(0.0, 1.0)
                                         for z in range(self._layers[1])]).reshape(1, self._layers[1])

        self._active_func_ptr = active_func_ptr
        if self._active_func_ptr is None:
            self._active_func_ptr = sigmoid

    def train(self, training_examples, training_annotations):

        # also called the 'hidden layer output matrix' computed as:
        # H is a n x m matrix (n training examples, m hidden nodes)
        # and H = [g(w_1 dot x_1 + b_1), ..., g(w_m dot x_1 + b_m)]
        #         [g(w_1 dot x_2 + b_1), ..., g(w_m dot x_2 + b_m)]
        #         [       ...            ...             ...      ]
        #         [g(w_1 dot x_n + b_1), ..., g(w_m dot x_n + b_m)]
        H = self._active_func_ptr(numpy.dot(training_examples, self._weights[0]) +
                                  self._hidden_bias)

        h_pseudoinverse = numpy.linalg.pinv(H)
        self._weights[-1] = numpy.dot(h_pseudoinverse, training_annotations)

    def classify(self, feature_vectors):
        # a = feature_vectors
        # a = self._active_func_ptr(numpy.dot(a, self._weights[0]) + self._hidden_bias)
        # a = numpy.dot(a, self._weights[-1])
        return numpy.dot(self._active_func_ptr(numpy.dot(feature_vectors, self._weights[0]) +
            self._hidden_bias), self._weights[-1])

