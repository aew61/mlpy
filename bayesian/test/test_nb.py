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
from stdlabels import *
from nb import *


class_label = boolean_label("lion")
feature_labels = [boolean_label("has-fur?"), boolean_label("long-teeth?"), boolean_label("scary?")]


class test_nb(unittest.TestCase):

    def test_constructor(self):
        classifier = nb(feature_labels, class_label)
        self.assertEqual(len(classifier._feature_nodes), 3)
        self.assertEqual(classifier._class_node._label.get_label(), "lion")
        for feature_name in ["has-fur?", "long-teeth?", "scary?"]:
            self.assertTrue(feature_name in classifier._feature_nodes.keys())

    def test_train(self):
        classifier = nb(feature_labels, class_label)

        dataset = [
            {"has-fur?": True, "long-teeth?": False, "scary?": False},
            {"has-fur?": False, "long-teeth?": True, "scary?": True},
            {"has-fur?": True, "long-teeth?": True, "scary?": True},
        ]
        annotations = [
            False,
            False,
            True,
        ]

        classifier.train(dataset, annotations)
        lion_node = classifier._class_node
        has_fur_node = classifier._feature_nodes["has-fur?"]
        long_teeth_node = classifier._feature_nodes["long-teeth?"]
        scary_node = classifier._feature_nodes["scary?"]

        conditions_table = numpy.array([[0.5, 0.5], [0.0, 1.0]], dtype=float)
        self.assertEqual(numpy.array([[0.67, 0.33]]).all(),
                         numpy.array([[numpy.round(x, 2) for x in lion_node._truth_table[0]]]).all())
        self.assertEqual(conditions_table.all(),
                         numpy.array([[numpy.round(y, 2) for y in x]
                                      for x in has_fur_node._truth_table]).all())
        self.assertEqual(conditions_table.all(),
                         numpy.array([[numpy.round(y, 2) for y in x]
                                      for x in long_teeth_node._truth_table]).all())
        self.assertEqual(conditions_table.all(),
                         numpy.array([[numpy.round(y, 2) for y in x]
                                      for x in scary_node._truth_table]).all())

    def _setup_lion_classifier(self):
        classifier = nb(feature_labels, class_label)
        classifier._class_node._truth_table = numpy.array([[0.9, 0.1]], dtype=float)
        classifier._feature_nodes["has-fur?"]._truth_table = numpy.array([[0.9, 0.1],
                                                                          [0.5, 0.5]], dtype=float)
        classifier._feature_nodes["long-teeth?"]._truth_table = numpy.array([[0.5, 0.5],
                                                                             [0.1, 0.9]], dtype=float)
        classifier._feature_nodes["scary?"]._truth_table = numpy.array([[0.5, 0.5],
                                                                        [0.2, 0.8]], dtype=float)
        return classifier

    def test_state_pr(self):
        classifier = self._setup_lion_classifier()

        feature_vector = {"has-fur?": True, "long-teeth?": False, "scary?": False}
        self.assertEqual(numpy.round(classifier.state_pr(feature_vector, True), 3), 0.001)
        self.assertEqual(numpy.round(classifier.state_pr(feature_vector, False), 4), 0.0225)

    def test_classify_pr(self):
        classifier = self._setup_lion_classifier()

        feature_vector = {"has-fur?": True, "long-teeth?": False, "scary?": False}
        expected_distribution = numpy.array([[0.0225, 0.001]], dtype=float)
        received_distribution = classifier.classify_pr(feature_vector)

        print("expected:%s, received: %s" % (expected_distribution, received_distribution))

        self.assertEqual(expected_distribution.all(), received_distribution.all())


if __name__ == "__main__":
    unittest.main()

