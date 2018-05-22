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


class elm(basenet.BaseNet):
    def __init__(self, layer_sizes, seed=None, afunc_ptrs=None, ignore_overflow=False):
        super(ELM, self).__init__(layer_sizes, seed=seed, afunc_ptrs=afunc_ptrs, ignore_overflow=ignore_overflow)
        assert(self.num_layers == 3)
        self.biases = self.biases[:-1]

    def _train(self, X, Y):
        old_settings = dict()
        if not self.ignore_overflow:
            old_settings = numpy.seterr(over="raise")
        else:
            old_settings = numpy.seterr(over="ignore")

        try:
            # also called the 'hidden layer output matrix' computed as:
            # H is a n x m matrix (n training examples, m hidden nodes)
            # and H = [g(w_1 dot x_1 + b_1), ..., g(w_m dot x_1 + b_m)]
            #         [g(w_1 dot x_2 + b_1), ..., g(w_m dot x_2 + b_m)]
            #         [       ...            ...             ...      ]
            #         [g(w_1 dot x_n + b_1), ..., g(w_m dot x_n + b_m)]
            H = self.afunc_ptr(numpy.dot(X, self.weights[0]) + self.biases[0])

            h_pseudoinverse = numpy.linalg.pinv(H)
            self.weights[-1] = numpy.dot(h_pseudoinverse, Y)
        except FloatingPointError:
            raise FloatingPointError("Overflow occured: please scale features")
        finally:
            numpy.seterr(**old_settings)

    """ """
    def predict(self, X):
        old_settings = dict()
        if not self.ignore_overflow:
            old_settings = numpy.seterr(over="raise")
        else:
            old_settings = numpy.seterr(over="ignore")

        try:
            # a = feature_vectors
            # a = self._active_func_ptr(numpy.dot(a, self._weights[0]) + self._hidden_bias)
            # a = numpy.dot(a, self._weights[-1])
            return numpy.dot(self.afunc_ptr(numpy.dot(X, self.weights[0]) +
                self.biases[0]), self.weights[-1])
        except:
            raise FloatingPointError("Overflow occured: please scale features")
        finally:
            numpy.seterr(**old_settings)
    """ """

