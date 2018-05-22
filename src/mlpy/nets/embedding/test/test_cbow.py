# SYSTEM IMPORTS
import numpy
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_emb_dir_ = os.path.join(_cd_, "..")
for _dir_ in [_cd_, _emb_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _emb_dir_
del _cd_


# PYTHON PROJECT IMPORTS
from cbow import cbow


def compute_one_hot_training(corpus, context_size):
    tokens = set()
    for l in corpus:
        tokens.update(l.split(" "))

    token_map = {w: i for i,w in enumerate(tokens)}

    contexts = list()
    output_tokens = list()

    for l in corpus:
        split_l = l.split(" ")
        for i, t in enumerate(split_l):
            if len(split_l[i-context_size:i]) == context_size:
                contexts.append(split_l[i-context_size:i])
                output_tokens.append(t)
            if len(split_l[i+1:i+1+context_size]) == context_size:
                contexts.append(split_l[i+1:i+1+context_size])
                output_tokens.append(t)

    # print
    # for l, o in zip(contexts, output_tokens):
    #     print("%s -> %s" % (l, o))

    context_corpus = list()
    output_vecs = list()
    for l, o in zip(contexts, output_tokens):
        assert(len(l) == context_size)

        one_hot_context = numpy.zeros(context_size * len(tokens))
        for i, t in enumerate(l):
            one_hot_context[i*context_size + token_map[t]] = 1

        one_hot_output = numpy.zeros(len(tokens))
        one_hot_output[token_map[o]] = 1

        context_corpus.append(one_hot_context)
        output_vecs.append(one_hot_output)

    assert(len(context_corpus) == len(output_vecs))
    s = context_corpus[0].shape
    # print(s)
    for l in context_corpus:
        # print(l.shape)
        assert(s == l.shape)

    # print(numpy.array(output_vecs))
    # print(numpy.array(context_corpus))

    return numpy.array(context_corpus), numpy.array(output_vecs)


def main():
    # corpus = ["hello how are you",
    #           "i am doing well",
    #          ]
    corpus = ["Hey this is sample corpus using only one context word"]
    num_embedding_dims = 5
    context_size = 2
    X, y = compute_one_hot_training(corpus, context_size)
    # print(X)
    # print()
    # print(y)

    num_epochs = 50
    m = cbow(y.shape[1], context_size, 3, learning_rate=0.05)

    for i in range(num_epochs):
        #print([w.shape for w in m.weights])
        # for w in m.weights:
            # print(w)
            # print()

        m.train(X, y)
        print(m.cost_function(X, y))


if __name__ == "__main__":
    main()

