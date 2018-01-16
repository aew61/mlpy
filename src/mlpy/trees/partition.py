# SYSTEM IMPORTS
# from abc import ABCMeta, abstractmethod
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


def create_partition(feature_index, feature_type, F, Y):
    if ftypes.is_discrete(feature_type):
        return DiscretePartition(feature_index, F, Y)
    elif ftypes.is_hierarchical(feature_type):
        return HierarchicalPartition(feature_index, F, Y)
    elif ftypes.is_continuous(feature_type):
        return ContinuousPartition(feature_index, F, Y)
    else:
        raise Exception("unknown feature type [%s] for feature [%s]" % (feature_type, feature_index))

"""
class PartitionBase(object): # metaclass=ABCMeta):
    def __init__(self, col_index, F, Y):
        self.col_index = col_index
        self.num_partition_values = 0

    # @abstractmethod
    # def test_example(self, x):
    #     pass

    def partition_data(self, X, Y):
        data_bin_indices = self.split_data_into_groups(X, Y)

        for i in range(self.num_partition_values - 1):
            # partition the data
            yield_indices = data_bin_indices == i
            # if numpy.sum(yield_indices) == 0:
            #     yield numpy.array([]), Y
            # else:
            yield X[yield_indices], Y[yield_indices]

        # now do the last one
        yield_indices = data_bin_indices == self.num_partition_values - 1
        if numpy.sum(yield_indices) == 0:
            yield numpy.array([]), Y
        else:
            yield X[yield_indices], Y[yield_indices]
"""

    # @abstractmethod
    # def split_data_into_groups(self, X, Y):
    #     pass

    # @abstractmethod
    # def create_partition_values(self, F, Y):
    #     pass


class ContinuousPartition(object): # PartitionBase):
    def __init__(self, col_index, F, Y):
        # super(ContinuousPartition, self).__init__(col_index, F, Y)
        self.col_index = col_index
        self.num_partition_values = 0
        self.le_partition_values = None
        self.lt_partition_value = None
        self.create_partition_values(F, Y)

    def test_example(self, x):
        out_index = numpy.argmax(x[self.col_index] <= self.le_partition_values)
        if out_index == self.num_partition_values - 2 and self.lt_partition_value is not None\
           and x[self.col_index] >= self.lt_partition_value:
                out_index = self.num_partition_values - 1
        return out_index

    """
    def split_data_into_groups(self, X, Y):
        data_bin_indices = numpy.argmax(X[:, self.col_index:self.col_index+1] <=
                                        self.le_partition_values, axis=1)

        if self.lt_partition_value is not None:
            # find elements that should be at the "<" condition
            lt_bools = X[data_bin_indices == self.num_partition_values - 2]\
                [:, self.col_index:self.col_index+1] >= self.lt_partition_value
            data_bin_indices[data_bin_indices == self.num_partition_values - 2][lt_bools] =\
                self.num_partition_values - 1
        return data_bin_indices
    """

    # partition values for continuous features can have 2 forms:
    # 1) p_1 <= p_2 <= p_3 ... <= p_n <= numpy.inf where numpy.inf is the "catch all" case
    # 2) p_1 <= p_2 <= p_3 ... <= p_n-1 < p_n <= numpy.inf. This only happens when
    #       p_n == max(F) because we want to make our grouping a little more fine-tuned.
    #       this case will use 2 member variables: self.le_partition_values & self.lt_partition_value.
    #       we know self.le_partition_values will ALWAYS have at least numpy.inf in it,
    #       and if self.lt_partition_value is not None, we just have to bump up
    #       the values with index == index(numpy.inf) + 1 and push the self.lt_partition_value
    #       indices in + self.num_partition_values - 2
    def create_partition_values(self, F, Y):
        assert(F.shape[0] == Y.shape[0])

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

        le_partition_values = numpy.unique((F[y_change_from_left] + F[y_change_from_right]) / 2)
        self.num_partition_values = le_partition_values.shape[0] + 1

        if le_partition_values[-1] == f_max:
            print("SETTING LT_PARTITION_VALUE")
            self.lt_partition_value = le_partition_values[-1]
            le_partition_values[-1] = numpy.inf
        else:
            self.lt_partition_value = None  # just making doubly sure (should be set in constructor)
            le_partition_values = numpy.append(le_partition_values, numpy.inf)
        self.le_partition_values = le_partition_values

    def partition_data(self, X, Y):
        # data_bin_indices = self.split_data_into_groups(X, Y)
        data_bin_indices = numpy.argmax(X[:, self.col_index:self.col_index+1] <=
                                        self.le_partition_values, axis=1)

        if self.lt_partition_value is not None:
            # find elements that should be at the "<" condition
            lt_bools = X[data_bin_indices == self.num_partition_values - 2]\
                [:, self.col_index:self.col_index+1] >= self.lt_partition_value
            data_bin_indices[data_bin_indices == self.num_partition_values - 2][lt_bools] =\
                self.num_partition_values - 1

        for i in range(self.num_partition_values - 1):
            # partition the data
            yield_indices = data_bin_indices == i
            # if numpy.sum(yield_indices) == 0:
            #     yield numpy.array([]), Y
            # else:
            yield X[yield_indices], Y[yield_indices]

        # now do the last one
        yield_indices = data_bin_indices == self.num_partition_values - 1
        if numpy.sum(yield_indices) == 0:
            yield numpy.array([]), Y
        else:
            yield X[yield_indices], Y[yield_indices]


class HierarchicalPartition(object): # PartitionBase):
    def __init__(self, col_index, F, Y):
        # super(HierarchicalPartition, self).__init__(col_index, F, Y)
        self.col_index = col_index
        self.num_partition_values = 0
        self.partition_values = None
        self.create_partition_values(F, Y)

    def test_example(self, x):
        out_index = numpy.argmax(x[self.col_index] == self.partition_values) - 1
        if out_index == -1:
            self.num_partition_values - 1
        return out_index

    """
    def split_data_into_groups(self, X, Y):
        data_bin_indices = numpy.argmax(X[:, self.col_index:self.col_index+1] ==
                                        self.partition_values, axis=1) - 1
        data_bin_indices[data_bin_indices == -1] = self.num_partition_values - 1
        return data_bin_indices
    """

    def create_partition_values(self, F, Y):
        vals = numpy.unique(F).astype(float)
        self.partition_values = numpy.insert(vals, 0, -numpy.inf)
        self.num_partition_values = self.partition_values.shape[0]

    def partition_data(self, X, Y):
        # data_bin_indices = self.split_data_into_groups(X, Y)
        data_bin_indices = numpy.argmax(X[:, self.col_index:self.col_index+1] ==
                                        self.partition_values, axis=1) - 1
        data_bin_indices[data_bin_indices == -1] = self.num_partition_values - 1

        for i in range(self.num_partition_values - 1):
            # partition the data
            yield_indices = data_bin_indices == i
            # if numpy.sum(yield_indices) == 0:
            #     yield numpy.array([]), Y
            # else:
            yield X[yield_indices], Y[yield_indices]

        # now do the last one
        yield_indices = data_bin_indices == self.num_partition_values - 1
        if numpy.sum(yield_indices) == 0:
            yield numpy.array([]), Y
        else:
            yield X[yield_indices], Y[yield_indices]


class DiscretePartition(object): # PartitionBase):
    def __init__(self, col_index, F, Y):
        # super(DiscretePartition, self).__init__(col_index, F, Y)
        self.col_index = col_index
        self.num_partition_values = 0
        self.partition_values = None
        self.create_partition_values(F, Y)

    def test_example(self, x):
        out_index = numpy.argmax(x[self.col_index] == self.partition_values) - 1
        if out_index == -1:
            out_index = self.num_partition_values - 1
        return out_index

    """
    def split_data_into_groups(self, X, Y):
        data_bin_indices = numpy.argmax(X[:, self.col_index:self.col_index+1] ==
                                        self.partition_values, axis=1) - 1
        data_bin_indices[data_bin_indices == -1] = self.num_partition_values - 1
        return data_bin_indices
    """

    def create_partition_values(self, F, Y):
        vals = numpy.unique(F).astype(float)
        self.partition_values = numpy.insert(vals, 0, -numpy.inf)
        self.num_partition_values = self.partition_values.shape[0]

    def partition_data(self, X, Y):
        # data_bin_indices = self.split_data_into_groups(X, Y)
        data_bin_indices = numpy.argmax(X[:, self.col_index:self.col_index+1] ==
                                        self.partition_values, axis=1) - 1
        data_bin_indices[data_bin_indices == -1] = self.num_partition_values - 1

        for i in range(self.num_partition_values - 1):
            # partition the data
            yield_indices = data_bin_indices == i
            # if numpy.sum(yield_indices) == 0:
            #     yield numpy.array([]), Y
            # else:
            yield X[yield_indices], Y[yield_indices]

        # now do the last one
        yield_indices = data_bin_indices == self.num_partition_values - 1
        if numpy.sum(yield_indices) == 0:
            yield numpy.array([]), Y
        else:
            yield X[yield_indices], Y[yield_indices]

"""
class Partition(object):
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

    def test_example(self, x):
        out_index = numpy.argmax(self.operator(x[self.feature_id], self.partition_values)) - 1
        if out_index == -1:
            out_index = self.partition_values.shape[0] - 2

        return out_index

    def partition_data(self, X, Y):
        data_bin_indices = numpy.argmax(self.operator(X[:, self.feature_id:self.feature_id+1],
                                                      self.partition_values), axis=1) - 1
        if ftypes.is_discrete(self.feature_type):
            data_bin_indices[data_bin_indices == -1] = self.partition_values.shape[0] - 2

        for i in range(self.partition_values.shape[0] - 2):
            # partition the data
            yield_indices = data_bin_indices == i
            yield X[yield_indices], Y[yield_indices]

        # now do the last one
        yield_indices = data_bin_indices == self.partition_values.shape[0] - 1
        if numpy.sum(yield_indices) == 0:
            yield numpy.array([]), Y
        else:
            yield X[yield_indices], Y[yield_indices]
"""

