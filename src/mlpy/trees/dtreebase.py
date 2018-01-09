# SYSTEM IMPORTS
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
import dtreenodedata
from data.features import ftypes


PURE_LABELS = -1


class DTreeBase(core.Base):
    def __init__(self, feature_header={}, max_depth=numpy.inf):
        self.feature_header = dict(feature_header)  # map data format indices -> feature type
        self.max_depth = max_depth

    def entropy(self, Y_f):
        unique_vals, counts = numpy.unique(Y_f, return_counts=True)
        pdf = counts / Y_f.shape[0]
        pdf[pdf == 0] = 2.0  # hax to make sure 0*log2(0) == 0...log2(2) == 0
        return -numpy.sum(pdf * numpy.log2(pdf))

    def information_gain(self, F, Y):
        unique_f_vals = numpy.unique(F)
        gain = 0.0
        for unique_f in unique_f_vals:
            gain -= (numpy.sum(F==unique_f) / F.shape[0]) * self.entropy(Y[F==unique_f])
        return gain

    def max_information_gain(self, X, Y, ignore_features=set()):
        max_f_gain = -numpy.inf
        max_f_index = PURE_LABELS

        f_gain = None  # preallocate variable

        unique_ys = numpy.unique(Y)

        # print("Y: %s\nunique_Y: %s" % (Y, unique_ys))

        if len(unique_ys) > 1:  # data is not "pure" i.e. more than one label type
            for i, f in enumerate(X.T):  # for each column (each column is a feature)
                if i not in ignore_features:
                    f_gain = self.information_gain(X[:,i], Y)
                    if f_gain > max_f_gain:
                        max_f_gain = f_gain
                        max_f_index = i
        return max_f_index, max_f_gain

    def get_nominal_partition_values(self, F, Y):
        return numpy.unique(F), numpy.min(F), numpy.max(F)

    def get_ordered_partition_values(self, F, Y):
        return numpy.unique(F), numpy.min(F), numpy.max(F)

    def get_continuous_partition_values(self, F, Y):
        assert(F.shape[0] == Y.shape[0])

        f_min = numpy.min(F)
        f_max = numpy.max(F)

        # need to find all unique Fs that split Y into different classes....
        # sort F (and Y) and then find the values of F where Y changes.

        # returns the indices of F that make F be in sorted order
        f_argsort = numpy.argsort(F)

        sorted_f = F[f_argsort]
        sorted_y = Y[f_argsort]

        # now find the values of F where Y changes
        y_change_from_left = numpy.zeros(F.shape[0], dtype=bool)
        y_change_from_right = numpy.zeros(F.shape[0], dtype=bool)
        diff = Y[:-1] != Y[1:]
        y_change_from_left[1:] = diff
        y_change_from_right[:-1] = diff

        # diff_f_values = F[y_change_indices]
        # print(diff_f_values)

        # partition_values = diff_f_values

        partition_values = numpy.unique((F[y_change_from_left] + F[y_change_from_right]) / 2)

        """
        if partition_values.shape[0] > 1:
            # find median between adjacent F values that have different classes
            # This should also be in sorted order seeing as we started with a sorted order
            partition_values = numpy.unique((diff_f_values[:-1] + diff_f_values[1:]) / 2)
        """

        # print(partition_values)

        return partition_values, f_min, f_max

    def get_hierarchical_partition_values(self, F, Y):
        return numpy.unique(F), numpy.min(F), numpy.max(F)

    def get_partition_values(self, f_index, F, Y):
        f_type = self.feature_header[f_index]
        func = self.get_nominal_partition_values

        if f_type == ftypes.ORDERED:        func = self.get_ordered_partition_values
        elif f_type == ftypes.CONTINUOUS:   func = self.get_continuous_partition_values
        elif f_type == ftypes.HIERARCHICAL: func = self.hierarchical_partition_values
        elif f_type != ftypes.NOMINAL:      raise Exception("feature [%s] (type [%s]) is not a recognized type" % (f_index, f_type))

        return func(F, Y)

