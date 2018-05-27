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
import partition


class id3tree(dtreebase.DTreeBase):
    def __init__(self, feature_header=None, max_depth=numpy.inf, use_gain_ratio=False):
        super(id3tree, self).__init__(feature_header=feature_header,
                                      max_depth=max_depth, use_gain_ratio=use_gain_ratio)
        self.tree_impl = core.dtypes.Tree()
        self.labels = list()

    def _check_tree(self):
        # at each node, make sure that the number of test functions == number of children
        for n in self.tree_impl.interiors():
            if len(n.data.test_functions) != len(n.children):
                print("node for feature [%s] of type [%s] has [%s] test funcs but [%s] children" %
                    (n.data.feature_id, n.data.feature_type,
                     len(n.data.test_functions), len(n.children)))
                raise Exception()

    def id3_training_algorithm(self, X, Y, parent, depth, ignored_features):
        self.num_nodes += 1
        # print("training algorithm")
        new_node = None
        if X.shape[0] > 0 and depth < self.max_depth:
            # choose the feature with max ig
            max_f_index, max_f_ig = self.max_information_gain(X, Y, ignored_features)
            if max_f_index != dtreebase.PURE_LABELS:
                #new_node = core.dtypes.Node(
                #    dtreenodedata.DTreeNodeData(max_f_index,
                #                                self.feature_header[max_f_index],
                #                                *self.get_partition_values(max_f_index,
                #                                                           X[:, max_f_index], Y)))
                new_node = core.dtypes.Node(
                    partition.create_partition(max_f_index, self.feature_header[max_f_index],
                                               X[:, max_f_index], Y))
            else:  # pure node choose majority class
                unique_ys = numpy.unique(Y, axis=0)
                new_node = core.dtypes.Node(unique_ys[0])

            # add node to tree
            if parent is None:
                self.tree_impl.root = new_node
            else:
                parent.children.append(new_node)

            if max_f_index != dtreebase.PURE_LABELS:
                # recursively call on partitioned data
                for new_X, new_Y in new_node.data.partition_data(X, Y):
                    # if new_X.shape[0] == 0:
                    #     print("feature [%s] of type [%s]" %
                    #         (max_f_index, self.feature_header[max_f_index]))
                    #     print(X.shape, Y.shape)
                    #     print(X[:, max_f_index])
                    #     print()

                    if new_Y.shape[0] == 0:
                        new_Y = Y
                    self.id3_training_algorithm(new_X, new_Y, new_node, depth+1,
                                                ignored_features|{max_f_index})
            
        else:
            # unique vals of Y with counts
            unique_ys, counts = numpy.unique(Y, axis=0, return_counts=True)
            if unique_ys.shape[0] == 0:
                print(X, Y)

            majority_y = unique_ys[numpy.argmax(counts)]
            new_node = core.dtypes.Node(majority_y)
            if parent is None:
                self.tree_impl.root = new_node
            else:
                parent.children.append(new_node)

    def _train(self, X, Y):
        self.id3_training_algorithm(X, Y, None, 1, set())
        # self._check_tree()

    def _predict_example(self, x):
        n = self.tree_impl.root
        if n is None:
            return None

        # go to the left child if the test is false, true otherwise
        while len(n.children) > 0:
            # test the current node
            n = n.children[n.data.test_example(x)]  # interior nodes: n.data is a func pointer, label otherwise
        return n.data

