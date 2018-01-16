# SYSTEM IMPORTS
import numpy
import operator
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
from features import ftypes


class LEPartition(object):
    def __init__(self, feature_id, feature_type, partition_values, min_f, max_f):
        self.feature_id = feature_id
        self.feature_type = feature_type

        # partition_values will have the form [-numpy.inf, <values>, numpy.inf]
        # the infs serve a purpose:
        #    1) -numpy.inf will make numpy.argmax(...) return a 0 if the feature val is not found
        #       for discrete features.
        #    2) numpy.inf will make numpy.argmax(...) spit out the correct index if the feature
        #       val is not found for continuous features.
        self.partition_values = numpy.sort(partition_values)
        self.min_f = min_f
        self.max_f = max_f
        self.operator = operator.le
        if ftypes.is_discrete(self.feature_type):
            self.operator = operator.eq
        self.num_partition_values = self.partition_values.shape[0] - 1
        # if ftypes.is_continuous(self.feature_type):
        #     self.partition_values = numpy.delete(self.partition_values, 0, axis=0)

    def test_example(self, x):
        out_index = numpy.argmax(self.operator(x[self.feature_id], self.partition_values)) - 1
        if out_index == -1:
            out_index = self.partition_values.shape[0] - 2
        else:
            out_index
        return out_index

    def partition_data(self, X, Y):
        data_bin_indices = numpy.argmax(self.operator(X[:, self.feature_id:self.feature_id+1],
                                                      self.partition_values), axis=1) - 1
        if ftypes.is_discrete(self.feature_type):
            data_bin_indices[data_bin_indices == -1] = self.partition_values.shape[0] - 2

        for i in range(self.num_partition_values - 1):
            # partition the data
            yield_indices = data_bin_indices == i
            yield X[yield_indices], Y[yield_indices]

        # now do the last one
        yield_indices = data_bin_indices == self.num_partition_values - 1
        if numpy.sum(yield_indices) == 0:
            yield numpy.array([]), Y
        else:
            yield X[yield_indices], Y[yield_indices]

