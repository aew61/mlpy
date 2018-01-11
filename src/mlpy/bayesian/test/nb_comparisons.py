import numpy
import operator
import os
import random
import re
import sys


_current_dir_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_current_dir_, "..", "..", "..")
_naive_bayes_classifier_dir_ = os.path.join(_current_dir_, "..", "..", "..", "naive-bayes-classifier")
_dirs_to_add_ = [_current_dir_, _src_dir_, _naive_bayes_classifier_dir_]
for _dir_ in _dirs_to_add_:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _dirs_to_add_
del _src_dir_
del _naive_bayes_classifier_dir_
del _current_dir_


# PYTHON PROJECT IMPORTS
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier

from mlpy.bayesian.nb import nb
from mlpy.bayesian.stdlabels import *
from mlpy.data_partitioning import abstract_partition


tokenization_string = "\"'().,!-<>/\\"


def create_dataset():
    data_toplevel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
        "examples", "datasets", "csmining", "parsed", "CSDMC2010_SPAM"))
    raw_data_path = os.path.join(data_toplevel_path, "TRAINING")  # includes validation data too

    examples = list()
    annotations = list()

    with open(os.path.join(data_toplevel_path, "SPAMTrain.label"), "r") as f:
        annotations = [' '.join(line.strip().split()).split() for line in f]

    text = None
    for data_file in os.listdir(raw_data_path):
        with open(os.path.join(raw_data_path, data_file), "r") as f:
            text = f.read()
        text = ' '.join([word for word in text.split() if len(word) > 3])
        examples.append((data_file, text))

    examples = sorted(examples, key=operator.itemgetter(0))
    annotations = sorted(annotations, key=operator.itemgetter(1))

    examples = [feature_vec for data_file, feature_vec in examples]
    annotations = [not bool(int(annotation)) for annotation, data_file in annotations]

    spam_examples = list()
    spam_annotations = list()
    ham_examples = list()
    ham_annotations = list()

    def partition_scheme(example, annotation):
        if annotation:
            spam_examples.append(example)
            spam_annotations.append(annotation)
        else:
            ham_examples.append(example)
            ham_annotations.append(annotation)
    abstract_partition(examples, annotations, partition_scheme)

    assert(len(ham_examples) + len(spam_examples) == len(examples))
    return (spam_examples, spam_annotations), (ham_examples, ham_annotations)


def partition_datasets(spam_dataset, ham_dataset):
    spam_examples, spam_annotations = spam_dataset
    ham_examples, ham_annotations = ham_dataset

    training_examples = list()
    training_annotations = list()

    validation_examples = list()
    validation_annotations = list()

    spam_in_training = list()

    random.seed(12345)
    percent_of_data_in_validation_set = 0.4
    def partition_scheme(example, annotation):
        if random.uniform(0, 1) <= percent_of_data_in_validation_set:
            validation_examples.append(example)
            validation_annotations.append(annotation)
        else:
            training_examples.append(example)
            training_annotations.append(annotation)
            if annotation:
                spam_in_training.append(1)

    abstract_partition(ham_examples, ham_annotations, partition_scheme)
    abstract_partition(spam_examples, spam_annotations, partition_scheme)

    spam_in_training = sum(spam_in_training)

    print("spam_in_training: %s, ham_in_training: %s" % (spam_in_training, len(training_examples) - spam_in_training))
    print("spam_in_validation: %s, ham_in_validation: %s" % (len(spam_examples) - spam_in_training, len(validation_examples) - len(spam_examples) + spam_in_training))

    return (training_examples, training_annotations),\
           (validation_examples, validation_annotations)


def tokenize(text):
    tokens = text.lower().split(" ")
    return [re.sub(str([tokenization_string]), "", token) for token in tokens]


def create_mlpy_nb_classifier(training_dataset):
    training_examples, training_annotations = training_dataset
    vocab = set()
    class_label = boolean_label("spam")

    parsed_training_examples = list()
    parsed_example = None
    tokens = None
    for example in training_examples:
        parsed_example = tokenize(example)
        vocab.update(parsed_example)
        parsed_training_examples.append(dict.fromkeys(parsed_example, True))

    # training_annotations = [not bool(annotation) for annotation in training_annotations]

    print("vocabulary length: %s" % len(vocab))
    feature_labels = [boolean_label(word) for word in vocab]
    c = nb(feature_labels, class_label)
    c.train(parsed_training_examples, training_annotations)
    return c


def create_nbc_nb_classifier(training_dataset):
    training_examples, training_annotations = training_dataset
    # training_annotations = [int(not bool(annotation)) for annotation in training_annotations]
    parsed_training_examples = [set(tokenize(example)) for example in training_examples]

    tr = Trainer(tokenizer.Tokenizer(stop_words=[], signs_to_remove=[tokenization_string]))
    for example, annotation in zip(parsed_training_examples, training_annotations):
        tr.train(example, annotation)

    print("number of tokens seen: %s" % len(tr.data.frequencies.keys()))
    return tr, Classifier(tr.data, tokenizer.Tokenizer(stop_words=[],
                                                       signs_to_remove=[tokenization_string]))


class mlpy_nb_handle(object):
    def __init__(self, nb):
        self._nb = nb

    def get_name(self):
        return "mlpy_nb"

    def get_class_counts(self):
        return self._nb._class_node._truth_table

    def pr_of_word(self, word):
        class_label = self._nb._class_node._label
        class_name = class_label.get_label()
        for class_state in class_label.get_all_states():
            print("Pr of word: [%s] for class state [%s]: %s" % (word, class_state, self._nb._feature_nodes[word].conditional_pr({class_name: class_state}, True)))
        return numpy.array([self._nb._feature_nodes[word].conditional_pr({class_name: state}, True) for state in class_label.get_all_states()])


class nbc_nb_handle(object):
    def __init__(self, nb):
        self._nb = nb

    def get_name(self):
        return "nbc_nb"

    def get_class_counts(self):
        return numpy.array([self._nb.data.getClassDocCount(class_label)
            for class_label in self._nb.data.getClasses()])

    def pr_of_word(self, word):
        # print("word: [%s] frequencies: %s" % (word, numpy.array([[self._nb.data.getFrequency(word, class_label) for class_label in self._nb.data.getClasses()]])))
        for class_state in self._nb.data.getClasses():
            print("Pr of word: [%s] for class state[%s]: %s" % (word, class_state, self._nb.getTokenProb(word, class_state)))
        return numpy.array([self._nb.getTokenProb(word, name) for name in self._nb.data.getClasses()])


def compare_class_counts(handles):
    comparison_vector = numpy.zeros((2, len(handles)), dtype=float)
    handle_names = list()
    index = 0
    for handle in handles:
        handle_names.append(handle.get_name())
        comparison_vector[:, index] = handle.get_class_counts()
        index += 1
    print("\nCOMPARING CLASS COUNTS")
    print("%s\n%s" % (handle_names, comparison_vector))


def compare_probabilities(classifier_handle_list, vocab):
    comparison_vector = numpy.zeros((2, len(classifier_handle_list)), dtype=float)
    handle_names = list()
    print("\nCOMPARING PROBABILITIES")
    for word in vocab[:1]:
        index = 0
        for classifier_handle in classifier_handle_list:
            handle_names.append(classifier_handle.get_name())
            comparison_vector[:, index] = classifier_handle.pr_of_word(word)
            index += 1

        print("%s\n%s" % (handle_names, comparison_vector))


def main():
    spam_dataset, ham_dataset = create_dataset()
    print("spam_dataset length: %s, ham_dataset length: %s" % (len(spam_dataset[0]), len(ham_dataset[0])))

    training_dataset, validation_dataset = partition_datasets(spam_dataset, ham_dataset)

    mlpy_nb = create_mlpy_nb_classifier(training_dataset)
    tr, nbc_nb = create_nbc_nb_classifier(training_dataset)

    vocab = nbc_nb.data.frequencies.keys()

    mlpy_handle = mlpy_nb_handle(mlpy_nb)
    nbc_handle = nbc_nb_handle(nbc_nb)
    handles = [mlpy_handle, nbc_handle]

    compare_class_counts(handles)
    compare_probabilities(handles, vocab)

if __name__ == "__main__":
    main()

