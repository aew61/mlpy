# SYSTEM IMPORTS
from abc import ABCMeta, abstractmethod


# PYTHON PROJECT IMPORTS


class abstractlabel(object):
    __metaclass__ = ABCMeta

    def __init__(self, label):
        self._label = label

    def get_label(self):
        return self._label

    @abstractmethod
    def hash(self, x):
        pass

    @abstractmethod
    def num_states(self):
        pass

    @abstractmethod
    def get_all_states(self, copy=False):
        pass

    def __str__(self):
        return "%s:%s" % (self.__class__.__name__, self.get_all_states())

    def __eq__(self):
        return (other is not None and isinstance(self, other.__class__)
            and isinstance(other, self.__class__))

