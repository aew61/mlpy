# SYSTEM IMPORTS
import os
import sys
import unittest


_current_dir_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_current_dir_, "..")
_dirs_to_add_ = [_current_dir_, _src_dir_]
for _dir_ in _dirs_to_add_:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _dirs_to_add_
del _src_dir_
del _current_dir_


# PYTHON PROJECT IMPORTS
from nbnode import *
from abstractlabel import *
from stdlabels import *


bool_label = boolean_label("Foo")
dis_label = discrete_label("Bar", ["foo", "bar", "baz"])

class test_nbnode(unittest.TestCase):

    def test_constructor(self):
        # initialize nbnode with no dependencies and a boolean label
        node = nbnode(bool_label)
        self.assertEqual(node._rows, 1)
        self.assertEqual(node._columns, 2)

        # initialize nbnode with no dependencies and a discrete non-boolean label
        node = nbnode(dis_label)
        self.assertEqual(node._rows, 1)
        self.assertEqual(node._columns, 3)

        # initialize nbnode with one dependency and a boolean label
        node = nbnode(bool_label, dependent_labels=[bool_label])
        self.assertEqual(node._rows, 2)
        self.assertEqual(node._columns, 2)

        # initialize nbnode with one dependency and a discrete non-boolean label
        node = nbnode(dis_label, dependent_labels=[dis_label])
        self.assertEqual(node._rows, 3)
        self.assertEqual(node._columns, 3)

        # initialize nbnode with multiple dependencies and a boolean label
        node = nbnode(bool_label, dependent_labels=[dis_label, bool_label])
        self.assertEqual(node._rows, 6)
        self.assertEqual(node._columns, 2)

        # initialize nbnode with multiple dependencies and a discrete non-boolean label
        node = nbnode(dis_label, dependent_labels=[dis_label, bool_label])
        self.assertEqual(node._rows, 6)
        self.assertEqual(node._columns, 3)

    def test_index_row(self):
        node = nbnode(bool_label)
        self.assertEqual(node.index_row({}), 0)  # no dependencies

        node = nbnode(bool_label, dependent_labels=[bool_label])
        self.assertEqual(node.index_row({"Foo": False}), 0)
        self.assertEqual(node.index_row({"Foo": True}), 1)

        node = nbnode(bool_label, dependent_labels=[dis_label])
        self.assertEqual(node.index_row({"Bar": "foo"}), 0)
        self.assertEqual(node.index_row({"Bar": "bar"}), 1)
        self.assertEqual(node.index_row({"Bar": "baz"}), 2)

        node = nbnode(bool_label, dependent_labels=[bool_label, dis_label])
        # we are sorting by label name and number of states, so this should be sorted
        # to [dis_label, bool_label]
        self.assertEqual(node.index_row({"Bar": "foo", "Foo": False}), 0)
        self.assertEqual(node.index_row({"Bar": "foo", "Foo": True}), 1)
        self.assertEqual(node.index_row({"Bar": "bar", "Foo": False}), 2)
        self.assertEqual(node.index_row({"Bar": "bar", "Foo": True}), 3)
        self.assertEqual(node.index_row({"Bar": "baz", "Foo": False}), 4)
        self.assertEqual(node.index_row({"Bar": "baz", "Foo": True}), 5)

    def test_conditional_pr(self):
        node = nbnode(bool_label)
        node._truth_table = numpy.array([[0.7, 0.3]])  # [[Pr(Foo=F), Pr(Foo=T)]]
        self.assertEqual(node.conditional_pr({}, False), 0.7)
        self.assertEqual(node.conditional_pr({}, True), 0.3)

        node = nbnode(bool_label, dependent_labels=[bool_label])
        node._truth_table = numpy.array([[0.4, 0.6],
                                         [0.03, 0.97]])
        self.assertEqual(node.conditional_pr({"Foo": False}, False), 0.4)
        self.assertEqual(node.conditional_pr({"Foo": False}, True), 0.6)
        self.assertEqual(node.conditional_pr({"Foo": True}, False), 0.03)
        self.assertEqual(node.conditional_pr({"Foo": True}, True), 0.97)


if __name__ == "__main__":
    unittest.main()

