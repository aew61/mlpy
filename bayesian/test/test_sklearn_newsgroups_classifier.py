# SYSTEM IMPORTS
import numpy
import os
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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


categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]


def filter_set(data):
    tokenization_string = "\"'().,!-<>/\\=?"
    def word_filter(example):
        return [x for x in re.sub(str([tokenization_string]), "", example.lower()).split() if len(x) > 3]

    for i in range(len(data)):
        data[i] = u"%s" % ' '.join(word_filter(data[i]))

    return data


def create_and_partition_dataset():
    training_set = fetch_20newsgroups(subset="train", categories=categories, shuffle=True,
                                      random_state=42)
    validation_set = fetch_20newsgroups(subset="test", categories=categories, shuffle=True,
                                        random_state=42)

    training_set.data = filter_set(training_set.data)
    validation_set.data = filter_set(validation_set.data)

    return (training_set.data, training_set.target),\
           (validation_set.data, validation_set.target)


def main():
    training_set, validation_set = create_and_partition_dataset()
    training_examples, training_annotations = training_set
    validation_examples, validation_annotations = validation_set

    count_vec = CountVectorizer(binary=True)
    training_examples = count_vec.fit_transform(training_examples)
    validation_examples = count_vec.transform(validation_examples)


    # training_examples.data = numpy.ones(len(training_examples.data))
    # validation_examples.data = numpy.ones(len(validation_examples.data))

    print(training_examples.shape)
    print(validation_examples.shape)
    # print(training_examples.shape)
    # print(training_examples[1, :])

    classifier = MultinomialNB().fit(training_examples, training_annotations)
    print(numpy.mean(validation_annotations == classifier.predict(validation_examples)))


if __name__ == "__main__":
    main()

