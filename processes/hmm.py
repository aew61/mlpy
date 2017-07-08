# SYSTEM IMPORTS
import numpy


# PYTHON PROJECT IMPORTS


class hmm(object):
    def __init__(self, scaling=True):
        self._scaling = scaling  # hmm probabilities are at risk of machine underflow
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

