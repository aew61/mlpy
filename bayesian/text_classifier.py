# SYSTEM IMPORTS
import os
import sys


_current_dir_ = os.path.abspath(os.path.dirname(__file__))
if _current_dir_ not in sys.path:
    sys.path.append(_current_dir_)
del _current_dir_


# PYTHON PROJECT IMPORTS
from nb import nb
from stdlabels import boolean_label, discrete_label


class TextClassifier(object):
    def __init__(self):
        self._vocabulary = set()
        self._classifier = None
        self._training_set = None
        self._word_filter_func_ptr = None

    def train(self, training_examples, training_annotations, class_label, word_filter_func_ptr):
        assert(isinstance(class_label, discrete_label))

        self._vocabulary = set()
        self._word_filter_func_ptr = word_filter_func_ptr
        filtered_examples = list()
        filtered_example = None
        for example in training_examples:
            # this assumes that word_filter_func_ptr will spit out a list of some kind
            # that contains the words from that example that the classifier will train with
            filtered_example = word_filter_func_ptr(example)
            self._vocabulary.update(filtered_example)
            filtered_examples.append(dict.fromkeys(filtered_example, 1.0))

        # print("len(filtered_examples): %s, len(annotations): %s" % (len(filtered_examples),
        #     len(training_annotations)))

        print("len(vocabulary): %s" % len(self._vocabulary))
        print("vocabulary: %s" % self._vocabulary)

        self._classifier = nb([boolean_label(x) for x in self._vocabulary], class_label)
        self._classifier.train(filtered_examples, training_annotations)

    def classify_text(self, examples):
        classifications = list()
        if isinstance(examples, basestring):
            classifications.append(
                self._classifier.classify_scores(
                    dict.fromkeys(self._word_filter_func_ptr(examples), 1.0)))
        else:
            for example in examples:
                classifications.append(
                    self._classifier.classify_scores(
                        dict.fromkeys(self._word_filter_func_ptr(example), 1.0)))
        return classifications

