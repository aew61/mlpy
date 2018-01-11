# SYSTEM IMPORTS
import numpy


# PYTHON PROJECT IMPORTS


class binary_svm(object):
    def __init__(self, kernel, class_error_rates):
        self._kernel = kernel
        self._class_error_rates = class_error_rates
        self._lagrange_multipliers = None
        self._support_vectors = None
        self._bias = 0.0

    def train(self, training_examples, training_annotations):
        num_examples, num_features = training_examples.shape
        assert(num_examples == len(training_annotations))

        # need to build 

    def _classify_example(self, feature_vector):
        classification = self._bias
        for i in range(len(self._support_vectors)):
            classification += self._lagrange_multipliers[i] *\
                self._ys[i] * self._kernel(self._support_vectors[i], feature_vector)
        return classification

    def classify(self, feature_vectors):
        classifications = numpy.zeros((len(feature_vectors), 1), dtype=float)

        for i in range(len(feature_vectors)):
            classifications[i, :] = self._classify_example(feature_vectors[i])
        return classifications

