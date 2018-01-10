# SYSTEM IMPORTS
from abc import ABCMeta, abstractmethod


# PYTHON PROJECT IMPORTS


class Base(metaclass=ABCMeta):
    def __init__(self):
        pass

    def train(self, X, y):
        self._train(X, y)
        return self

    def predict(self, X):
        return [self._predict_example(x) for x in X]

    def predict_generator(self, X):
        for x in X:
            yield self._predict_example(x)

    @abstractmethod
    def _train(self, X, y):
        pass

    @abstractmethod
    def _predict_example(self, x):
        return None

