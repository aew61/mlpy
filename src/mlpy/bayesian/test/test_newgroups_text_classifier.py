# SYSTEM IMPORTS
import numpy
import os
import re
from sklearn.datasets import fetch_20newsgroups
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
from mlpy.bayesian.stdlabels import discrete_label


categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]


def create_and_partition_dataset():
    training_set = fetch_20newsgroups(subset="train", categories=categories, shuffle=True,
                                      random_state=42)
    validation_set = fetch_20newsgroups(subset="test", categories=categories, shuffle=True,
                                        random_state=42)

    return (training_set.data, training_set.target),\
           (validation_set.data, validation_set.target)


def main():
    training_set, validation_set = create_and_partition_dataset()
    training_examples, training_annotations = training_set
    validation_examples, validation_annotations = validation_set
    class_label = discrete_label("type", [1, 2, 3, 4])

    tokenization_string = "[\"'\|_().,!-<>/\\=\?"
    def word_filter(example):
        return [x for x in re.sub(str([tokenization_string]), "", example.lower()).split() if len(x) > 3]

    classifier = text_classifier()
    classifier.train(training_examples, training_annotations, class_label, word_filter)

    # for word in classifier._vocabulary:
    #     print(word)

    def get_max_score(scores):
        return class_label.get_all_states()[numpy.argmax(scores)]

    output_scores = classifier.classify_text(validation_examples)

    output_classifications = numpy.zeros(len(output_scores))
    for i in range(len(output_scores)):
        output_classifications[i] = get_max_score(output_scores[i])
    print(numpy.mean(output_classifications == validation_annotations))


if __name__ == "__main__":
    main()

