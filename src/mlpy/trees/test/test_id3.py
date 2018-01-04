# SYSTEM IMPORTS
import numpy
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_cd_, "..")
_module_dir_ = os.path.join(_src_dir_, "..")
for _dir_ in [_cd_, _src_dir_, _module_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _module_dir_
del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import id3
from features import ftypes


def test_ig(d, X, Y):
    for i, col in enumerate(X.T):
        print("ig for feature {0}: {1:.5f}".format(i, d.information_gain(X[:,i], Y)))
    print("max f: {0}: {1:.5f}".format(*d.max_information_gain(X, Y)))


def test_train(d, X, Y):
    d.train(X, Y)


def main():

    """"""
    X = numpy.array([[1, 0, 0],
                     [0, 1, 1],
                     [1, 0, 1],
                     [0, 1, 1]], dtype=float)
    Y = numpy.array([[0],
                     [0],
                     [1],
                     [1]], dtype=float)

    feature_header = {0: ftypes.NOMINAL, 1: ftypes.NOMINAL, 2: ftypes.NOMINAL}
    """"""

    """
    X = numpy.array([[0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [1],
                     [1],
                     [1],
                     [0],
                     [0],
                     [1],
                     [1],
                     [1]], dtype=float)
    Y = numpy.array([[1],
                     [1],
                     [1],
                     [1],
                     [1],
                     [1],
                     [1],
                     [1],
                     [1],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0]], dtype=float)

    feature_header = {0: ftypes.NOMINAL}
    """

    d = id3.DTree(feature_header=feature_header)

    test_ig(d, X, Y)
    print()
    print()
    test_train(d, X, Y)
    return d


if __name__ == "__main__":
    main()

