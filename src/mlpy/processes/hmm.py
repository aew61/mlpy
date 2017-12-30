# SYSTEM IMPORTS
import numpy
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_cd_, "..")
_dirs_to_add_ = [_cd_, _src_dir_]
for _dir_ in _dirs_to_add_:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _dirs_to_add_
del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import core


class hmm(core.Base):
    def __init__(self): # , scaling=True):
        # self._scaling = scaling  # hmm probabilities are at risk of machine underflow
        self._transition_matrix = None
        self._emmision_matrix = None
        self._initial_probabilities = None  # will be a column vector

    def forward(self, observations):
        cache = 0.0
        return cache

    def backwards(self, data):
        pass

    def viterbi_log(self, data):
        pass

    def baum_welch(self, data):
        pass

    def _train(self, X, y):
        pass

    def _predict_example(self, x):
        return None

