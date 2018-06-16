# SYSTEM IMPORTS
from abc import ABCMeta, abstractmethod
import numpy


# PYTHON PROJECT IMPORTS


class Base(metaclass=ABCMeta):
    def __init__(self):
        pass

    def train(self, X, y, *args, **kwargs):
        self._train(X, y, *args, **kwargs)
        return self

    def predict(self, X):
        first_out = self._predict_example(X[0])
        out = numpy.zeros((X.shape[0], 1), dtype=type(first_out))
        out[0] = first_out
        for i, x in enumerate(X[1:]):
            out[i] = self._predict_example(x)
        return out
        # return [self._predict_example(x) for x in X]

    def predict_generator(self, X):
        for x in X:
            yield self._predict_example(x)

    @abstractmethod
    def _train(self, X, y, *args, **kwargs):
        pass

    @abstractmethod
    def _predict_example(self, x):
        return None

