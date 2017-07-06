# SYSTEM IMPORTS
import numpy
import os
import sys


_current_dir_ = os.path.abspath(os.path.dirname(__file__))
if _current_dir_ not in sys.path:
    sys.path.append(_current_dir_)
del _current_dir_


# PYTHON PROJECT IMPORTS
from abstractlabel import *

class continuous_label(abstractlabel):
    def __init__(self, label):
        super(continuous_label, self).__init__(label)

    def hash(self, x):
        return x

    def num_state(self):
        return numpy.inf

    def get_all_states(self, copy=False):
        return numpy.inf


class discrete_label(abstractlabel):
    def __init__(self, label, states):
        super(discrete_label, self).__init__(label)
        self._states = states

    def hash(self, x):
        return self._states.index(x) if x in self._states else -1

    def num_states(self):
        return len(self._states)

    def get_all_states(self, copy=False):
        return list(self._states) if copy else self._states


class nominal_label(abstractlabel):
    def __init__(self, label, states):
        super(nominal_label, self).__init__(self)
        self._states = states

    def hash(self, x):
        return self._states.index(x) if x in self._states else -1

    def num_states(self):
        return len(self._states)

    def get_all_states(self, copy=False):
        return list(self._states) if copy else self._states

class boolean_label(discrete_label):
    def __init__(self, label):
        super(boolean_label, self).__init__(label, [False, True])


class text_label(boolean_label):
    def __init__(self, label):
        super(text_label, self).__init__(label)
