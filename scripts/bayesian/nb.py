# SYSTEM IMPORTS
import numpy
import random


# PYTHON PROJECT IMPORTS
from nbnode import nbnode


class nb(object):
    def __init__(self, feature_labels, classes_label, smoothing_coeff=0.0):
        self._smoothing_coeff = smoothing_coeff
        self._feature_nodes = {
            feature.get_label(): nbnode(feature, dependent_labels=[classes_label],
                                        smoothing_coeff=smoothing_coeff)
            for feature in feature_labels
        }
        self._class_node = nbnode(classes_label, smoothing_coeff=smoothing_coeff)
        self._default_pr = 0.0

    def train(self, training_examples, training_annotations):
        assert(len(training_examples) == len(training_annotations))

        node = None
        index = None

        max_words_in_training_examples = 0

        # print("\nTraining")
        # for every example, count the number of times that that feature was seen by 1
        for example, annotation in zip(training_examples, training_annotations):
            # print("training_example: %s, annotation: %s" % (example, annotation))
            if len(example) > max_words_in_training_examples:
                max_words_in_training_examples = len(example)
            for feature_name, feature_value in example.items():
                # get the node for this feature
                node = self._feature_nodes[feature_name]
                # get the index of the row for this annotation in the feature node's truth table
                index = node.index_row({self._class_node._label.get_label(): annotation})
                # increment the count by 1
                node._truth_table[index][node._label.hash(feature_value)] += 1
            index = self._class_node.index_row({})
            self._class_node._truth_table[index][self._class_node._label.hash(annotation)] += 1


        # print("\nNormalizing")
        # go through and convert everything into probabilities i.e. divide by frequency
        self._default_pr = self._smoothing_coeff / (self._smoothing_coeff *
            (max_words_in_training_examples + 1)) if self._smoothing_coeff != 0.0 else 0.0
        smoothed_label_counts = self._class_node._truth_table.T +\
            self._smoothing_coeff * (max_words_in_training_examples + 1)
        for feature_name, feature_node in self._feature_nodes.items():
            # for every feature that this node has, we want to divide it by the number of times
            # that class label appeared in our training data. We can do this VERY fast using
            # numpy (whew)
            # print("%s truth table:\n%s" % (feature_name, feature_node._truth_table))
            feature_node._truth_table /= smoothed_label_counts

        self._class_node._truth_table /= float(numpy.sum(self._class_node._truth_table))
        # print("%s truth table:\n%s" % (self._class_node._label.get_label(),
        #                                self._class_node._truth_table))



    def state_pr(self, feature_vector_dict, class_value):
        prob = self._class_node.conditional_pr({}, class_value,
                                               default_pr=self._default_pr)
        for feature_name, feature_node in self._feature_nodes.items():
            prob *= feature_node.conditional_pr({self._class_node._label.get_label(): class_value},
                                                feature_vector_dict[feature_name],
                                                default_pr=self._default_pr)
        return prob

    def state_score(self, feature_vector_dict, class_value):
        score = numpy.log(self._class_node.conditional_pr({}, class_value,
                                                          default_pr=self._default_pr))
        for feature_name, feature_node in self._feature_nodes.items():
            score +=\
                numpy.log(feature_node.conditional_pr({self._class_node._label.get_label(): class_value},
                                                      feature_vector_dict[feature_name],
                                                      default_pr=self._default_pr))
        return score

    def classify_pr(self, feature_vector_dict):
        return numpy.array([[self.state_pr(feature_vector_dict, class_state)
                             for class_state in self._class_node._label.get_all_states()]],
                           dtype=float)

    def classify_scores(self, feature_vector_dict):
        return numpy.array([[self.state_score(feature_vector_dict, class_state)
                             for class_state in self._class_node._label.get_all_states()]],
                           dtype=float)

