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

    def default_state(self):
        return 0.0

    def get_all_states(self, copy=False):
        return numpy.inf


class discrete_label(abstractlabel):
    def __init__(self, label, states, default_state=None):
        super(discrete_label, self).__init__(label)
        self._states = states
        assert((default_state is None or default_state in states) and len(states) > 0)
        self._default_state = default_state if default_state is not None else states[0]

    def hash(self, x):
        return self._states.index(x) if x in self._states else -1

    def num_states(self):
        return len(self._states)

    def default_state(self):
        return self._default_state

    def get_all_states(self, copy=False):
        return list(self._states) if copy else self._states


class nominal_label(abstractlabel):
    def __init__(self, label, states, default_state=None):
        super(nominal_label, self).__init__(self)
        self._states = states
        assert((default_state is None or default_state in states) and len(states) > 0)
        self._default_state = default_state if default_state is not None else states[0]

    def hash(self, x):
        return self._states.index(x) if x in self._states else -1

    def num_states(self):
        return len(self._states)

    def default_state(self):
        return self._default_state

    def get_all_states(self, copy=False):
        return list(self._states) if copy else self._states

class boolean_label(discrete_label):
    def __init__(self, label):
        super(boolean_label, self).__init__(label, [False, True])


class text_label(boolean_label):
    def __init__(self, label):
        super(text_label, self).__init__(label)

