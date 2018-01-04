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
import dtreebase
import dtreenodedata


class DTree(dtreebase.DTreeBase):
    def __init__(self, feature_header={}):
        super(DTree, self).__init__(feature_header=feature_header)
        self.tree_impl = core.dtypes.Tree()
        self.labels = list()

    def id3_training_algorithm(self, X, Y, parent):
        pass

    def _train(self, X, Y):
        unique_labels = numpy.unique(Y)
        for unique_label in unique_labels:
            self.labels.append(core.dtypes.Node(unique_label))

        self.id3_training_algorithm(X, Y, None)

    def _predict_example(self, x):
        n = self.tree_impl.root

        if n is None:
            return None

        # go to the left child if the test is false, true otherwise
        while len(n.children) > 0:
            # test the current node
            n = n.children[n.data.test_example(x)]  # interior nodes: n.data is a func pointer, label otherwise
        return n.data

