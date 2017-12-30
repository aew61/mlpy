# SYSTEM IMPORTS
import numpy
import operator
import os
import random
import re
import sys


_current_dir_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_current_dir_, "..", "..", "..")
_dirs_to_add_ = [_current_dir_, _src_dir_]
for _dir_ in _dirs_to_add_:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _dirs_to_add_
del _src_dir_
del _current_dir_


# PYTHON PROJECT IMPORTS
from mlpy.bayesian import text_classifier
from mlpy.data_partitioning import abstract_partition
from mlpy.bayesian.stdlabels import boolean_label


def create_and_partition_dataset():
    data_toplevel_path = os.path.join("/scratch", "madeline", "CSDMC2010_SPAM")
    raw_data_path = os.path.join(data_toplevel_path, "TRAINING")  # includes validation data too

    examples = list()
    annotations = list()

    ### Read in data set ###

    with open(os.path.join(data_toplevel_path, "SPAMTrain.label"), "r") as f:
        annotations = [' '.join(line.strip().split()).split() for line in f]

    text = None
    for data_file in os.listdir(raw_data_path):
        with open(os.path.join(raw_data_path, data_file), "r") as f:
            text = f.read()
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

    ### partition it into training and validation sets ###

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


def main():
    training_set, validation_set = create_and_partition_dataset()

    training_examples, training_annotations = training_set
    validation_examples, validation_annotations = validation_set

    class_label = boolean_label("spam")

    tokenization_string = "\"'().,!-<>/\\=?"
    def word_filter(example):
        return [x for x in re.sub(str([tokenization_string]), "", example.lower()).split() if len(x) > 3]

    classifier = text_classifier()
    classifier.train(training_examples, training_annotations, class_label, word_filter)

    correct = 0
    incorrect = 0

    output_scores = classifier.classify_text(validation_examples)
    assert(len(validation_annotations) == len(output_scores))

    progress = 0
    total = len(validation_annotations)
    threshold = 0

    def get_max_score(scores):
        return class_label.get_all_states()[numpy.argmax(scores)]


    for scores, annotation in zip(output_scores, validation_annotations):
        if (float(progress) / float(total)) * 100 >= threshold:
            print("%s%%" % int((float(progress) / total) * 100)),
            if correct + incorrect > 0:
                print("%s correct" % (float(correct) / (float(correct + incorrect))))
            else:
                print("")
            threshold += 5

        # print("max_score: %s, annotation: %s" % (get_max_score(output_scores), annotation))
        if get_max_score(scores) == annotation:
            correct += 1
        else:
            incorrect += 1
        progress += 1

    print("percent correctly classified: %s" % (float(correct) / len(validation_set[0])))

if __name__ == "__main__":
    random.seed(12345)
    main()

