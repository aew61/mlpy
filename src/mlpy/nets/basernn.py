# SYSTEM IMPORTS
from abc import ABCMeta, abstractmethod
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
import core


class BaseRNN(core.Base, metaclass=ABCMeta):
    def __init__(self, input_size, output_size, bptt_truncate=4, afuncs=None, afunc_primes=None, seed=None):
        super(BaseRNN, self).__init__()
        self.afuncs = afuncs
        self.afunc_primes = afunc_primes
        numpy.random.seed(seed)
        self.input_size = input_size
        self.output_size = output_size
        self.bptt_truncate = bptt_truncate

    @abstractmethod
    def compute_layer(self, X):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def back_propagate_through_time(self, X, Y):
        pass

    def _predict_example(self, x):
        return self.predict(x.reshape(1, x.shape[0]))

    def feed_forward(self, X):
        Os = numpy.zeros((X.shape[0], self.output_size))
        for i in range(X.shape[0]):  # for each example
            Os[i] = self.compute_layer(X[i])
        return Os

    def predict(self, X):
        assert(X.shape[1] == self.input_size)
        Os = self.feed_forward(X)
        max_indices = numpy.argmax(Os, axis=1)
        Os[:, :] = 0
        Os[range(Os.shape[0]), max_indices] = 1
        return Os

    def loss_function(self, X, Y):
        L = 0
        N = 0
        num_examples = len(X)
        for i in range(num_examples):
            self.reset()
            N += X[i].shape[0]
            Os = self.predict_proba(X[i])
            L += -1*numpy.sum(numpy.log(Os[range(X[i].shape[0]), numpy.argmax(Y[i], axis=1)]))
        self.reset()
        return L/N

    def predict_proba(self, X):
        assert(X.shape[1] == self.input_size)
        return self.feed_forward(X)

