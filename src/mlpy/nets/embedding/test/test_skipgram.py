# SYSTEM IMPORTS
import matplotlib.pyplot as plt
import numpy
import os
from sklearn.manifold import TSNE
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_emb_dir_ = os.path.join(_cd_, "..")
_src_dir_ = os.path.join(_cd_, "..", "..", "..")
for _dir_ in [_cd_, _emb_dir_, _src_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _src_dir_
del _emb_dir_
del _cd_


# PYTHON PROJECT IMPORTS
from data.text import contok
from skipgram import skipgram


def main():
    # corpus = ["hello how are you",
    #           "i am doing well",
    #          ]
    # corpus = ["Hey this is sample corpus using only one context word"]
    corpus = ["the cat sat on the mat",
              "the cat slept on the mat",
             ]
    num_embedding_dims = 10
    context_size = 2
    tok = contok(context_size).tokenize(corpus)
    y, X = tok.transform(corpus)

    num_epochs = 1000
    m = skipgram(X.shape[1], context_size, 3, learning_rate=0.01)

    losses = numpy.zeros(num_epochs)

    for i in range(num_epochs):
        #print([w.shape for w in m.weights])
        # for w in m.weights:
            # print(w)
            # print()

        m.train(X, y)
        losses[i] = m.cost_function(X, y)
    plt.plot(numpy.arange(num_epochs), losses)
    plt.show()


if __name__ == "__main__":
    main()

