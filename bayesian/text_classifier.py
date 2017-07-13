# SYSTEM IMPORTS
import numpy
import os
from sklearn.feature_extraction.text import CountVectorizer
import sys


_current_dir_ = os.path.abspath(os.path.dirname(__file__))
if _current_dir_ not in sys.path:
    sys.path.append(_current_dir_)
del _current_dir_


# PYTHON PROJECT IMPORTS
from nb import nb
from stdlabels import boolean_label, discrete_label


class text_classifier(object):
    def __init__(self):
        self._vocabulary = set()
        self._classifier = None
        self._training_set = None
        self._word_filter_func_ptr = None
        self._count_vectorizer = CountVectorizer(binary=True)

    def train_2(self, training_examples, training_annotations, class_label):
        assert(isinstance(class_label, discrete_label))

        self._count_vectorizer.fit_transform(training_examples)
        self._vocabulary = set(self._count_vectorizer.vocabulary_.keys())
        feature_vectors = list()
        analyze_func = self._count_vectorizer.build_analyzer()
        for example in training_examples:
            feature_vectors.append(dict.fromkeys(analyze_func(example), True))

        self._classifier = nb([boolean_label(x) for x in self._vocabulary], class_label)
        self._classifier.train(feature_vectors, training_annotations)

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
            filtered_examples.append(dict.fromkeys(filtered_example, True))

        # print("len(filtered_examples): %s, len(annotations): %s" % (len(filtered_examples),
        #     len(training_annotations)))

        print("len(vocabulary): %s" % len(self._vocabulary))
        # print("vocabulary: %s" % self._vocabulary)

        self._classifier = nb([boolean_label(x) for x in self._vocabulary], class_label)
        self._classifier.train(filtered_examples, training_annotations)

    def classify_text_2(self, examples):
        classifications = numpy.zeros((len(examples), self._classifier._class_node._label.num_states()), dtype=float)
        index = 0
        analyzer_func = self._count_vectorizer.build_analyzer()
        for example in examples:
            classifications[index, :] =\
                self._classifier.classify_scores(
                    dict.fromkeys(analyzer_func(example), True))
        return classifications

    def classify_text(self, examples):
        classifications = None
        if isinstance(examples, basestring):
            classifications =\
                self._classifier.classify_scores(
                    dict.fromkeys(self._word_filter_func_ptr(examples), True))
        else:
            classifications = numpy.zeros((len(examples),
                self._classifier._class_node._label.num_states()), dtype=float)
            index = 0
            for example in examples:
                classifications[index, :] =\
                    self._classifier.classify_scores(
                        dict.fromkeys(self._word_filter_func_ptr(example), True))
                index += 1
        return classifications

