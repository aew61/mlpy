# SYSTEM IMPORTS
import collections
import numpy
import os
from scipy.sparse import csr_matrix
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_net_dir_ = os.path.join(_cd_, "..")
_mlpy_dir_ = os.path.join(_cd_, "..", "..")
for _dir_ in [_cd_, _net_dir_, _mlpy_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _mlpy_dir_
del _net_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import rnn
from data import unsupervised


def load_data():
    cd = os.path.abspath(os.path.dirname(__file__))
    test_data_dir = os.path.join(cd, "test_data", "lm")
    train_data = list()
    dev_data = list()
    test_data = list()

    with open(os.path.join(test_data_dir, "train"), "r") as f:
        for line in f:
            train_data.append(line.strip().rstrip().split())
    with open(os.path.join(test_data_dir, "dev"), "r") as f:
        for line in f:
            dev_data.append(line.strip().rstrip().split())
    with open(os.path.join(test_data_dir, "test"), "r") as f:
        for line in f:
            test_data.append(line.strip().rstrip().split())

    return train_data, dev_data, test_data


def parse_data(vocab, data):
    new_data = list()
    for l in data:
        new_l = list(["<s>"])
        for w in l:
            if w in l:
                new_l.append(w)
            else:
                new_l.append("<unk>")
        new_data.append(new_l + ["</s>"])
    return new_data


def prune_words(train, dev, test, prune_occurances_lt=1):
    c = collections.Counter()
    for l in train:
        c.update(l)
    inverse_map = collections.defaultdict(list)
    for w, count in c.items():
        inverse_map[count].append(w)

    if prune_occurances_lt < 0:
        prune_occurances_lt = 0
    for c in range(prune_occurances_lt):
        if c in inverse_map:
            del inverse_map[c]
    vocab = set(["<unk>", "<s>", "</s>"])
    for l in inverse_map.values():
        vocab.update(l)
    return vocab, parse_data(vocab, train), parse_data(vocab, dev), parse_data(vocab, test)


def make_sequence_vectorized(vocab_dict, sequence):
    vec_sequence = numpy.zeros((len(sequence), len(vocab_dict)))
    for i, w in enumerate(sequence):
        if w not in vocab_dict:
            vec_sequence[i][vocab_dict["<unk>"]] = 1
        else:        
            vec_sequence[i][vocab_dict[w]] = 1
    return csr_matrix(vec_sequence[:-1, :]), csr_matrix(vec_sequence[1:, :])


def make_vectorized(train, dev, test, vocab, cutoff=numpy.inf):
    vocab_dict = {w: i for i, w in enumerate(vocab)}
    X_train = list()
    Y_train = list()
    X_dev = list()
    Y_dev = list()
    X_test= list()
    Y_test = list()

    for i, s in enumerate(train):
        if i < cutoff:
            X_, Y_ = make_sequence_vectorized(vocab_dict, s)
            X_train.append(X_)
            Y_train.append(Y_)
    for i, s in enumerate(dev):
        if i < cutoff:
            X_, Y_ = make_sequence_vectorized(vocab_dict, s)
            X_dev.append(X_)
            Y_dev.append(Y_)
    for i, s in enumerate(test):
        if i < cutoff:
            X_, Y_ = make_sequence_vectorized(vocab_dict, s)
            X_test.append(X_)
            Y_test.append(Y_)
    return (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test)


def main():
    prune_occurance_lt = 15
    cutoff = 100 # numpy.inf # 1000
    hidden_size = 100
    num_epochs = 5

    print("loading data", end="", flush=True)
    train, dev, test = load_data()
    print("done.")
    print("pruning words with occurances < %s..." % prune_occurance_lt, end="", flush=True)
    vocab, train, dev, test = prune_words(train, dev, test, prune_occurances_lt=prune_occurance_lt)
    print("done. vocab size: %s" % len(vocab))
    print("vectorizing data with cutoff: %s..." % cutoff, end="", flush=True)
    train, dev, test = make_vectorized(train, dev, test, vocab, cutoff=cutoff)
    print("done.")
    print("training model for %s epoch(s)..." % num_epochs)

    X_train, Y_train = train
    X_dev, Y_dev = dev
    X_test, Y_test = test

    lm = rnn.rnn(len(vocab), len(vocab), hidden_size=hidden_size, seed=10).train(X_train, Y_train,
                                                                                 verbose=2,
                                                                                 epochs=num_epochs)
    acc = 0
    tot = 0
    for X_t, Y_t in zip(X_test, Y_test):
        Y_hat = lm.predict(X_t)
        assert(Y_hat.shape == Y_t.shape)
        # print(numpy.argmax(Y_hat, axis=1))
        # print(numpy.argmax(Y_t.toarray(), axis=1))
        acc += numpy.sum(numpy.argmax(Y_hat, axis=1) == numpy.argmax(Y_t.toarray(), axis=1))
        tot += Y_t.shape[0]
    print("accuracy on test data: %s" % (float(acc) / tot))


if __name__ == "__main__":
    main()

