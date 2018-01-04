# SYSTEM IMPORTS
import functools
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
from features import ftypes


def base_test_func(partition_val, comparison_func, feature_id, x):
    print(x)
    return comparison_func(x[feature_id], partition_val)


def base_partition_func(partition_val, comparison_func, feature_id, X):
    print(X)
    return comparison_func(X[:, feature_id], partition_val)


class DTreeNodeData(object):
    def __init__(self, feature_id, feature_type, partition_values):
        self.feature_id = feature_id
        self.feature_type = feature_type
        self.partition_values = numpy.sort(partition_values)  # guarantee that they're sorted
        self.partition_functions = list()
        self.test_functions = list()

        self.create_partition_functions()

    def create_partition_functions(self):
        # we know that the partition_values are sorted...but enforce that here
        partition_values = numpy.sort(self.partition_values)
        if self.feature_type == ftypes.NOMINAL:
            for p in partition_values:
                self.partition_functions.append(functools.partial(base_partition_func,
                    p, operator.eq, self.feature_id))
                self.test_functions.append(functools.partial(base_test_func,
                    p, operator.eq, self.feature_id))
        
        elif self.feature_type == ftypes.ORDERED:
            for p in partition_values:
                self.partition_functions.append(functools.partial(base_partition_func,
                    p, operator.eq, self.feature_id))
                self.test_functions.append(functools.partial(base_test_func,
                    p, operator.eq, self.feature_id))

        elif self.feature_type == ftypes.CONTINUOUS:
            for p in partition_values[:-1]:
                self.partition_functions.append(functools.partial(base_partition_func,
                    p, operator.le, self.feature_id))
                self.test_functions.append(functools.partial(base_test_func,
                    p, operator.le, self.feature_id))
            self.partition_functions.append(functools.partial(base_partition_func,
                partition_values[-2], operator.gt, self.feature_id))
            self.test_functions.append(functools.partial(base_test_func,
                partition_values[-2], operator.gt, self.feature_id))

        elif self.feature_type == ftypes.HIERARCHICAL:
            for p in partition_values:
                self.partition_functions.append(lambda val: True)
                self.test_functions.append(lambda val: 0)

        else:
            raise Exception("feature type [%s] is an unrecognized type" % self.feature_type)

    def test_example(self, x):
        for i, pf in enumerate(self.partition_functions):
            if pf(numpy.array([x])):
                return i
        raise Exception("feature [%s] of type [%s] did not pass a partition function for example [%s]" %
            (self.feature_id, self.feature_type, x))

    def partition_data(self, X, Y):
        new_X = X
        new_Y = Y

        print()
        print("partition_data")
        #print("x.shape: %s, y.shape: %s" % (new_X.shape, new_Y.shape))
        #print("X: %s" % new_X)
        #print("Y: %s" % new_Y)
        print("feature type: %s" % self.feature_type)

        for pf in self.partition_functions:
            pf_X = pf(new_X)
            print("\t\t%s" % pf_X)
            # print(new_X)
            # print(new_Y)

            print("\tyielding data X: %s" % new_X[pf_X])
            print("\tyielding data Y: %s" % new_Y[pf_X])

            yield new_X[pf_X], new_Y[pf_X]
            new_X = new_X[numpy.logical_not(pf_X)]
            new_Y = new_Y[numpy.logical_not(pf_X)]

