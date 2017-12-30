# SYSTEM IMPORTS
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
imoprt dtreebase


class DTree(dtreebase.DTreeBase):
    def __init__(self):
        self.tree_impl = core.dtypes.Tree()
        self.labels = list()

    def _train(self, X, Y):
        unique_labels = numpy.unique(y, axis=0)
        for unique_label in unique_labels:
            self.labels.append(core.dtypes.Node(unique_label))

        # order the features in X (columns are separate features) by their information gain

    def _predict_example(self, x):
        n = self.tree_impl.root

        if n is None:
            return None

        # go to the left child if the test is false, true otherwise
        while len(n.children) > 0:
            # test the current node
            n = n.children[n.data(x)]  # interior nodes: n.data is a func pointer, label otherwise
        return n.data

