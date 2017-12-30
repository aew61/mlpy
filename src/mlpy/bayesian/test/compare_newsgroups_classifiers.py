# SYSTEM IMPORTS
import copy
import numpy
import os
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sys

_current_dir_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_current_dir_, "..", "..", "..")
_naive_bayes_classifier_dir_ = os.path.join(_src_dir_, "naive-bayes-classifier")
_dirs_to_add_ = [_current_dir_, _src_dir_, _naive_bayes_classifier_dir_]
for _dir_ in _dirs_to_add_:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _dirs_to_add_
del _naive_bayes_classifier_dir_
del _src_dir_
del _current_dir_


# PYTHON PROJECT IMPORTS
from mlpy.bayesian import text_classifier
from mlpy.bayesian.stdlabels import discrete_label

from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier


categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
tokenization_string = "[\"'\|_().,!-<>/\\=\?"
def word_filter(example):
    return [x for x in re.sub(str([tokenization_string]), "", example.lower()).split() if len(x) > 3]


def filter_data(data):
    new_data = list()
    for i in data:
        new_data.append(u"%s" % ' '.join(word_filter(i)))
    return new_data


def create_and_partition_dataset():
    training_set = fetch_20newsgroups(subset="train", categories=categories, shuffle=True,
                                      random_state=42)
    validation_set = fetch_20newsgroups(subset="test", categories=categories, shuffle=True,
                                        random_state=42)

    return (training_set.data, training_set.target),\
           (validation_set.data, validation_set.target)


# def convert_examples_to_text_classifier_using_sklearn(count_vectorizer, examples):
#     feature_vectors = list()
#     analyze_func = count_vectorizer.build_analyzer()
#     for example in examples:
#         feature_vectors.append(dict.fromkeys(analyze_func(example), 1.0))
#     return feature_vectors


def create_text_classifier(training_examples, training_annotations):
    print("creating text classifier")
    class_label = discrete_label("type", categories)
    # since actual annotations are in range of [1, 4] (all integers)
    # then to convert them to the appropriate label would be to subtract 1 from each
    training_annotations = [categories[x] for x in training_annotations]

    classifier = text_classifier()
    classifier.train(training_examples, training_annotations, class_label, word_filter)
    print("\t->done")
    return classifier


def create_sklearn_classifier(training_examples, training_annotations):
    print("creating sklearn classifier")
    examples = filter_data(training_examples)
    count_vec = CountVectorizer(binary=True)

    examples = count_vec.fit_transform(examples)
    classifier = MultinomialNB().fit(examples, training_annotations)
    print("\t->done")
    return count_vec, classifier


def create_naive_bayes_classifier(training_examples, training_annotations):
    print("creating naive bayes classifier")
    annotations = [categories[x] for x in training_annotations]

    news_trainer = Trainer(tokenizer.Tokenizer(stop_words=[], signs_to_remove=[tokenization_string]))
    for example, annotation in zip(training_examples, annotations):
        news_trainer.train(example, annotation)
    classifier = Classifier(news_trainer.data, tokenizer.Tokenizer(stop_words=[], signs_to_remove=[tokenization_string]))
    print("\t->done")
    return classifier


def test_text_classifier(classifier, validation_examples, validation_annotations):
    print("testing text classifier")
    annotations = [categories[x] for x in validation_annotations]

    output_scores = list()
    for output_dist in classifier.classify_text(validation_examples):
        output_scores.append(categories[numpy.argmax(output_dist)])

    print("\t->done")
    return numpy.mean(numpy.array(annotations) == numpy.array(output_scores))


def test_sklearn_classifier(count_vec, classifier, validation_examples, validation_annotations):
    print("testing sklearn classifier")
    examples = count_vec.transform(validation_examples)

    output_scores = classifier.predict(examples)
    print("\t->done")
    return numpy.mean(validation_annotations == output_scores)


def test_naive_bayes_classifier(classifier, validation_examples, validation_annotations):
    print("testing naive bayes classifier")
    annotations = [categories[x] for x in validation_annotations]

    output_scores = list()
    for example in validation_examples:
        output_scores.append(classifier.classify(example)[0][0])
    # print(output_scores)
    print("\t->done")
    return numpy.mean(numpy.array(annotations) == numpy.array(output_scores))
 

def main():

    ## making training a validation sets
    training_set, validation_set = create_and_partition_dataset()
    training_examples, training_annotations = training_set
    validation_examples, validation_annotations = validation_set

    text_classifier = create_text_classifier(copy.deepcopy(training_examples),
                                             copy.deepcopy(training_annotations))
    count_vec, sklearn_classifier = create_sklearn_classifier(copy.deepcopy(training_examples),
                                                              copy.deepcopy(training_annotations))
    nb_classifier = create_naive_bayes_classifier(copy.deepcopy(training_examples),
                                                  copy.deepcopy(training_annotations))

    sklearn_accuracy = test_sklearn_classifier(count_vec, sklearn_classifier,
                                               copy.deepcopy(validation_examples),
                                               copy.deepcopy(validation_annotations))
    text_classifier_accuracy = test_text_classifier(text_classifier,
                                                    copy.deepcopy(validation_examples),
                                                    copy.deepcopy(validation_annotations))
    nb_accuracy = test_naive_bayes_classifier(nb_classifier,
                                              copy.deepcopy(validation_examples),
                                              copy.deepcopy(validation_annotations))
    print("text_classifier accuracy: %s" % text_classifier_accuracy)
    print("sklearn accuracy: %s" % sklearn_accuracy)
    print("nb_accuracy: %s" % nb_accuracy)


if __name__ == "__main__":
    main()

