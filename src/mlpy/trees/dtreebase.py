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
import data.feature_types as ftypes


PURE_LABELS = -1


class DTreeBase(core.Base):
    def __init__(self, feature_header=None, max_depth=numpy.inf, use_gain_ratio=False):
        if feature_header is None:
            feature_header = dict()

        self.feature_header = dict(feature_header)  # map data format indices -> feature type
        self.max_depth = max_depth
        self.use_gain_ratio = use_gain_ratio
        self.num_nodes = 0

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

        if len(unique_ys) > 1:  # data is not "pure" i.e. more than one label type
            for i, f in enumerate(X.T):  # for each column (each column is a feature)
                if i not in ignore_features:
                    f_gain = self.information_gain(X[:, i], Y)

                    if self.use_gain_ratio:
                        f_gain /= self.entropy(X[:, i])

                    if f_gain > max_f_gain:
                        max_f_gain = f_gain
                        max_f_index = i
        return max_f_index, max_f_gain
       

    def get_hierarchical_partition_values(self, F, Y):
        return self.get_discrete_partition_values(F, Y)

