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


def parse_corpus(vocab, corpus):
    X_data = list()
    Y_data = list()
    for l in corpus:
        new_l = list(["<s>"])
        for w in l:
            if w in vocab:
                new_l.append(w)
            else:
                new_l.append("<unk>")
        new_l += ["</s>"]
        X_data.append(new_l[:-1])
        Y_data.append(new_l[1:])
    return X_data, Y_data


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
    return vocab, parse_corpus(vocab, train), parse_corpus(vocab, dev), parse_corpus(vocab, test)


def cutoff_data(train, dev, test, cutoff=numpy.inf):
    if cutoff != numpy.inf:
        return (train[0][:cutoff], train[1][:cutoff],), (dev[0][:cutoff], dev[1][:cutoff],),\
               (test[0][:cutoff], test[1][:cutoff],)
    return train, dev, test


def make_sequence_vectorized(vocab_dict, sequence):
    vec_sequence = numpy.zeros((len(sequence), len(vocab_dict)))
    for i, w in enumerate(sequence):
        if w not in vocab_dict:
            vec_sequence[i][vocab_dict["<unk>"]] = 1
        else:        
            vec_sequence[i][vocab_dict[w]] = 1
    return csr_matrix(vec_sequence)


def make_vectorized(train, dev, test, vocab):
    vocab_dict = {w: i for i, w in enumerate(vocab)}
    X_train = list()
    Y_train = list()
    X_dev = list()
    Y_dev = list()
    X_test= list()
    Y_test = list()

    X_tr, Y_tr = train
    X_d, Y_d = dev
    X_te, Y_te = test

    for sx, sy in zip(X_tr, Y_tr):
        X_ = make_sequence_vectorized(vocab_dict, sx)
        Y_ = make_sequence_vectorized(vocab_dict, sy)
        assert(X_.shape == Y_.shape)
        X_train.append(X_)
        Y_train.append(Y_)
    for sx, sy in zip(X_d, Y_d):
        X_ = make_sequence_vectorized(vocab_dict, sx)
        Y_ = make_sequence_vectorized(vocab_dict, sy)
        assert(X_.shape == Y_.shape)
        X_dev.append(X_)
        Y_dev.append(Y_)
    for sx, sy in zip(X_te, Y_te):
        X_ = make_sequence_vectorized(vocab_dict, sx)
        Y_ = make_sequence_vectorized(vocab_dict, sy)
        assert(X_.shape == Y_.shape)
        X_test.append(X_)
        Y_test.append(Y_)
    return (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test)


def check_inverse_indices(train_ws, dev_ws, test_ws, train_is, dev_is, test_is, vocab):
    vocab_dict = {w: i for i, w in enumerate(vocab)}
    inv_vocab_dict = {i: w for i, w in enumerate(vocab)}

    def check_data(data_ws, data_is):
        # print(len(data_ws), len(data_is))
        assert(len(data_ws) == len(data_is))
        for s_ws, s_is in zip(data_ws, data_is):
            assert(len(s_ws) == len(s_is))
            for w, i in zip(s_ws, s_is):
                assert(vocab_dict[w] == i)
                assert(inv_vocab_dict[i] == w)
    check_data(train_ws[0], train_is[0])
    check_data(train_ws[1], train_is[1])
    check_data(dev_ws[0], dev_is[0])
    check_data(dev_ws[1], dev_is[1])
    check_data(test_ws[0], test_is[0])
    check_data(test_ws[1], test_is[1])

def convert_sequence_to_vocab_indices(s, vocab_dict):
    s_new = list()
    for w in s:
        if w not in vocab_dict:
            s_new.append(vocab_dict["<unk>"])
        else:
            s_new.append(vocab_dict[w])
    return s_new


def convert_corpus_to_vocab_indices(corpus, vocab_dict):
    X_c, Y_c = corpus

    X_new = list()
    Y_new = list()
    for i, s in enumerate(X_c):
        X_new.append(convert_sequence_to_vocab_indices(s, vocab_dict))
    for i, s in enumerate(Y_c):
        Y_new.append(convert_sequence_to_vocab_indices(s, vocab_dict))
    return X_new, Y_new


def convert_all_data_to_vocab_indices(train, dev, test, vocab):
    vocab_dict = {w: i for i, w in enumerate(vocab)}
    new_corpi = list()
    for corpus in [train, dev, test]:
        new_corpi.append(convert_corpus_to_vocab_indices(corpus, vocab_dict))
    return tuple(new_corpi)


class VGIterator(object):
    def __init__(self, obj):
        self.obj = obj
        self.current = 0

    def __next__(self):
        if self.current >= len(self.obj):
            raise StopIteration
        else:
            arr = self.obj[self.current]
            self.current += 1
            return arr

class VG(object):
    def __init__(self, corpus, vocab_size):
        self.corpus = corpus
        self.vocab_size = vocab_size

    def __iter__(self):
        return VGIterator(self)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        arr = numpy.zeros((len(self.corpus[index]), self.vocab_size))
        arr[numpy.arange(len(self.corpus[index])), self.corpus[index]] = 1
        return arr


def check_data_format(csr_format, vg_format):
    for A_csr_i, A_is_i in zip(csr_format, vg_format):
            # print(type(A_csr_i), type(A_is_i))
            # print(A_csr_i)
            A_csr_i_arr = A_csr_i.toarray()
            assert(A_csr_i_arr.shape == A_is_i.shape)
            is_ = A_csr_i_arr != A_is_i
            if numpy.sum(is_) > 0:
                raise Exception("Data format error! Arrays contain different elements:\n%s\n%s" %
                    (A_csr_i_arr[is_], A_is_i[is_]))


def test_model(model, X_test, Y_test):
    acc = 0
    tot = 0
    for X_t, Y_t in zip(X_test, Y_test):
        Y_hat = model.predict(X_t)
        assert(Y_hat.shape == Y_t.shape)
        # print(numpy.argmax(Y_hat, axis=1))
        # print(numpy.argmax(Y_t.toarray(), axis=1))
        if not isinstance(Y_t, numpy.ndarray):
            acc += numpy.sum(numpy.argmax(Y_hat, axis=1) == numpy.argmax(Y_t.toarray(), axis=1))
        else:
            acc += numpy.sum(numpy.argmax(Y_hat, axis=1) == numpy.argmax(Y_t, axis=1))
        tot += Y_t.shape[0]
    return float(acc) / tot


def main():
    prune_occurance_lt = 15
    cutoff = numpy.inf # 1000
    hidden_size = 100
    num_epochs = 5

    print("loading data...", end="", flush=True)
    train, dev, test = load_data()
    print("done.")
    print("pruning words with occurances < %s..." % prune_occurance_lt, end="", flush=True)
    vocab, train, dev, test = prune_words(train, dev, test, prune_occurances_lt=prune_occurance_lt)
    vocab_size = len(vocab)

    train, dev, test = cutoff_data(train, dev, test, cutoff=cutoff)

    print("done. vocab size: %s" % len(vocab))
    print("vectorizing data with cutoff: %s..." % cutoff, end="", flush=True)
    # train_csr, dev_csr, test_csr = make_vectorized(train, dev, test, vocab)
    train_is, dev_is, test_is = convert_all_data_to_vocab_indices(train, dev, test, vocab)
    print("done.")
    print("Checking data formats...")

    """
    check_inverse_indices(train, dev, test, train_is, dev_is, test_is, vocab)

    X_train_csr, Y_train_csr = train_csr
    X_dev_csr, Y_dev_csr = dev_csr
    X_test_csr, Y_test_csr = test_csr
    """

    X_train_is, Y_train_is = VG(train_is[0], vocab_size), VG(train_is[1], vocab_size)
    X_dev_is, Y_dev_is = VG(dev_is[0], vocab_size), VG(dev_is[1], vocab_size)
    X_test_is, Y_test_is = VG(test_is[0], vocab_size), VG(test_is[1], vocab_size)

    """
    check_data_format(X_train_csr, X_train_is)
    check_data_format(Y_train_csr, Y_train_is)
    check_data_format(X_dev_csr, X_dev_is)
    check_data_format(Y_dev_csr, Y_dev_is)
    check_data_format(X_test_csr, X_test_is)
    check_data_format(Y_test_csr, Y_test_is)
    """
    print("done.")

    # print("training csr model for %s epoch(s)..." % num_epochs)
    # lm_csr = rnn.rnn(len(vocab), len(vocab), hidden_size=hidden_size, seed=10)
    # lm_csr.train(X_train_csr, Y_train_csr, verbose=2, epochs=num_epochs)

    print("training vg model for %s epoch(s)..." % num_epochs)
    lm_vg = rnn.rnn(len(vocab), len(vocab), hidden_size=hidden_size, seed=10)
    lm_vg.train(X_train_is, Y_train_is, verbose=2, epochs=num_epochs)

    # acc_csr = test_model(lm_csr, X_test_csr, Y_test_csr)
    acc_vg = test_model(lm_vg, X_test_is, Y_test_is)
    # print("csr acc: {0:.3f}".format(acc_csr))
    print("vg acc: {0:.3f}".format(acc_vg))

if __name__ == "__main__":
    main()

