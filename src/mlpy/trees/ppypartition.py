# SYSTEM IMPORTS
import functools
import inspect
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
del _dir_


# PYTHON PROJECT IMPORTS
from data.features import ftypes


def base_test_func(partition_val, comparison_func, feature_id, x):
    # print(x)
    return comparison_func(x[feature_id], partition_val)


def base_partition_func(partition_val, comparison_func, feature_id, X):
    # print(X)
    # print("partitioning on feature value: %s" % partition_val)
    out = comparison_func(X[:, feature_id], partition_val)
    if numpy.sum(out) == 0:
        print("out: %s" % X[:, feature_id])
    return out


class PPYPartition(object):
    def __init__(self, feature_id, feature_type, partition_values, f_min, f_max):
        self.feature_id = feature_id
        self.feature_type = feature_type
        self.partition_functions = list()
        self.test_functions = list()
        self.f_min = f_min
        self.f_max = f_max

        sorted_p_value_indices = numpy.argsort(partition_values)
        self.partition_values = partition_values[sorted_p_value_indices]
        # if counts is not None:
        #     self.counts = counts[sorted_p_value_indices]

        self.create_partition_functions()
        if ftypes.is_discrete(self.feature_type):
            assert(len(self.partition_functions) == len(self.test_functions) - 1)

    def _create_continuous_funcs_max_val(self, max_p_value):
        main_op = operator.le
        extra_op = operator.gt

        if max_p_value == self.f_max:
            # then swap the two (max value IS the max so there is nothing > than it)
            main_op = operator.lt
            extra_op = operator.ge

        for op in [main_op, extra_op]:
            self.partition_functions.append(functools.partial(base_partition_func,
                max_p_value, op, self.feature_id))
            self.test_functions.append(functools.partial(base_test_func,
                max_p_value, op, self.feature_id))

    def _create_continuous_funcs(self, sorted_p_values):
        if sorted_p_values.shape[0] == 0:
            raise Exception("Cannot partition feature with no partition values!")

        for p in sorted_p_values[:-1]:
            self.partition_functions.append(functools.partial(base_partition_func,
                p, operator.le, self.feature_id))
            self.test_functions.append(functools.partial(base_test_func,
                p, operator.le, self.feature_id))
        self._create_continuous_funcs_max_val(sorted_p_values[-1])


    def create_partition_functions(self):
        # we know that the partition_values are sorted...but enforce that here
        partition_values = numpy.sort(self.partition_values)
        if ftypes.is_discrete(self.feature_type):
            for p in partition_values:
                self.partition_functions.append(functools.partial(base_partition_func,
                    p, operator.eq, self.feature_id))
                self.test_functions.append(functools.partial(base_test_func,
                    p, operator.eq, self.feature_id))

            # this function is needed because if the feature is discrete, it is possible
            # for a feature value to not be in the training data. This is avoidable if
            # we include all feature values in the header, but this approach produces
            # the same results by creating a subchild that is pure with the majority class
            # viewed up at this node.
            self.test_functions.append(lambda val: True)

        elif ftypes.is_continuous(self.feature_type):
            # so complex it justifies a separate function
            self._create_continuous_funcs(partition_values)

        elif ftypes.is_hierarchical(self.feature_type):
            for p in partition_values:
                self.partition_functions.append(lambda val: True)
                self.test_functions.append(lambda val: 0)

        else:
            raise Exception("feature type [%s] is an unrecognized type" % self.feature_type)

    def test_example(self, x):
        for i, pf in enumerate(self.test_functions):
            if pf(x):
                return i
        raise Exception("feature [%s] of type [%s] did not pass a partition function with value [%s]" %
            (self.feature_id, self.feature_type, x[self.feature_id]))

    def partition_data(self, X, Y):
        new_X = X
        new_Y = Y

        for pf in self.partition_functions:
            pf_X = pf(new_X)

            # yield_X = new_X[pf_X]
            # yield_Y = new_Y[pf_X]

            # if yield_X.shape[0] == 0:
            #     print("feature [%s] of type [%s] no data? p_vals %s" %
            #         (self.feature_id, self.feature_type, self.partition_values))
            #     print("source:")
            #     print("".join(inspect.getsourcelines(pf.func)[0]), pf.args)
            #     print(numpy.sum(new_X <= self.partition_values[i]))

            yield new_X[pf_X], new_Y[pf_X]

            if not ftypes.is_continuous(self.feature_type):
                new_X = new_X[numpy.logical_not(pf_X)]
                new_Y = new_Y[numpy.logical_not(pf_X)]

        if ftypes.is_discrete(self.feature_type):
            # return data that will create a "pure" node with the majority class
            # majority_class = numpy.argmax(self.counts)
            unique_ys, counts = numpy.unique(Y, return_counts=True)
            # if unique_ys.shape[0] == 0:
            #     print(X, Y)

            # print("yielding no X for feature [%s] of type [%s]. unique_ys: %s, counts: %s" %
            #     (self.feature_id, self.feature_type, unique_ys, counts))
            yield numpy.array([]), Y[Y == unique_ys[numpy.argmax(counts)]]

