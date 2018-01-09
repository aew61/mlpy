# SYSTEM IMPORTS
from abc import ABCMeta, abstractmethod


# PYTHON PROJECT IMPORTS


class Base(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def _train(self, X, y):
        pass

    def train(self, X, y):
        self._train(X, y)
        return self

    @abstractmethod
    def _predict_example(self, x):
        return None

    def predict(self, X):
        for x in X:
            yield self._predict_example(x)

