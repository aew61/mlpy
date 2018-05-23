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
from contok import contok
from skipgram import skipgram


def main():
    # corpus = ["hello how are you",
    #           "i am doing well",
    #          ]
    corpus = ["Hey this is sample corpus using only one context word"]
    # corpus = ["hi"]
    num_embedding_dims = 5
    context_size = 2
    tok = contok(context_size).tokenize(corpus)
    y, X = tok.transform(corpus)

    num_epochs = 100
    m = skipgram(X.shape[1], context_size, 3, learning_rate=0.1)

    for i in range(num_epochs):
        #print([w.shape for w in m.weights])
        # for w in m.weights:
            # print(w)
            # print()

        m.train(X, y)
        print(m.cost_function(X, y))


if __name__ == "__main__":
    main()

