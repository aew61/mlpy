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
    one_hot_corpus = list()
    for l in corpus:
        one_hot_l = list()
        for t in l.split(" "):
            one_hot_w = numpy.zeros(len(tokens))
            one_hot_w[token_map[t]] = 1
            one_hot_l.append(one_hot_w)
        one_hot_corpus.append(one_hot_l)

    # now go through and, using the context size, compute the corpus
    context_corpus = list()
    output_vecs = list()
    for l in one_hot_corpus:
        for i in range(context_size, len(l)-1, context_size):
            context = l[i-context_size:i]
            out_vec = l[i]

            context_corpus.append(numpy.concatenate(tuple(context)))
            output_vecs.append(out_vec)

    return numpy.array(context_corpus), numpy.array(output_vecs)


def main():
    corpus = ["hello how are you",
              "i am doing well",
             ]

    context_size = 2
    X, y = compute_one_hot_training(corpus, context_size)
    print(X)
    print()
    print(y)
    


if __name__ == "__main__":
    main()

