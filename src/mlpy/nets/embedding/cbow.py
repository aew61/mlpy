# SYSTEM IMPORTS
import numpy
import os
import sys


_cd_ = os.path.dirname(os.path.abspath(__file__))
_net_dir_ = os.path.join(_cd_, "..")
_src_dir_ = os.path.join(_cd_, "..", "..")
for _dir_ in [_cd_, _net_dir_, _src_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _net_dir_
del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import activation_functions as af
import ann


class cbow(ann.ann):
    def __init__(self, vocab_size, context_size, num_embedding_dims, learning_rate=1.0):
        super(cbow, self).__init__([vocab_size, num_embedding_dims, vocab_size],
                                   afuncs=[af.linear, af.softmax],
                                   afunc_primes=[af.linear_prime, af.softmax_prime],
                                   learning_rate=learning_rate, weight_decay_coeff=0.0)
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.biases = [numpy.zeros(b.shape) for b in self.biases]


    def avg_context(self, X):
        return numpy.mean([X[:, i*self.vocab_size:(i+1)*self.vocab_size]
                          for i in range(self.context_size)], axis=0)

    def cost_function(self, X, y):
        _, as_ = self.complete_feed_forward(X)
        # z = zs[-1]
        # return numpy.sum(numpy.log(numpy.sum(numpy.exp(z), axis=1))) - numpy.sum(z[y == 1])
        return -numpy.sum(numpy.log(as_[-1][y==1]))

    def feed_forward(self, X):
        X_mean = self.avg_context(X)
        return super(cbow, self).feed_forward(X_mean)

    def complete_feed_forward(self, X):
        zs = list()

        X_mean = self.avg_context(X)
        as_ = list([X_mean])
        a = X_mean
        z = None
        for afunc, weight in zip(self.afuncs, self.weights):
            z = numpy.dot(a, weight)
            zs.append(z)
            a = afunc(z)
            as_.append(a)
        return zs, as_

    def _predict_example(self, x):
        return self.weights[-1].T[x == 1]

