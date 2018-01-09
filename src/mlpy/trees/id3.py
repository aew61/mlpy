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


class ID3DTree(dtreebase.DTreeBase):
    def __init__(self, feature_header={}, max_depth=numpy.inf):
        super(ID3DTree, self).__init__(feature_header=feature_header, max_depth=max_depth)
        self.tree_impl = core.dtypes.Tree()
        self.labels = list()

    def id3_training_algorithm(self, X, Y, parent, depth, ignored_features=set()):
        # print("training algorithm")
        new_node = None
        if X.shape[0] > 0 and depth < self.max_depth:
            # choose the feature with max ig
            max_f_index, max_f_ig = self.max_information_gain(X, Y, ignored_features)
            if max_f_index != dtreebase.PURE_LABELS:
                new_node = core.dtypes.Node(
                    dtreenodedata.DTreeNodeData(max_f_index,
                                                self.feature_header[max_f_index],
                                                *self.get_partition_values(max_f_index,
                                                                           X[:, max_f_index], Y)))
            else:  # pure node choose majority class
                unique_ys = numpy.unique(Y)
                new_node = core.dtypes.Node(unique_ys[0])

            # add node to tree
            if parent is None:
                self.tree_impl.root = new_node
            else:
                parent.children.append(new_node)

            if max_f_index != dtreebase.PURE_LABELS:
                # recursively call on partitioned data
                for new_X, new_Y in new_node.data.partition_data(X, Y):
                    self.id3_training_algorithm(new_X, new_Y, new_node, depth+1,
                                                ignored_features=ignored_features|{max_f_index})
            
        else:
            # unique vals of Y with counts
            unique_ys, counts = numpy.unique(Y, return_counts=True)
            majority_y = unique_ys[numpy.argmax(counts)]
            new_node = core.dtypes.Node(majority_y)
            if parent is None:
                self.tree_impl.root = new_node
            else:
                parent.children.append(new_node)

    def _train(self, X, Y):
        self.id3_training_algorithm(X, Y, None, 1)

    def _predict_example(self, x):
        n = self.tree_impl.root
        if n is None:
            return None

        # go to the left child if the test is false, true otherwise
        while len(n.children) > 0:
            # test the current node
            n = n.children[n.data.test_example(x)]  # interior nodes: n.data is a func pointer, label otherwise
        return n.data

