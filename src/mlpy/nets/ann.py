# SYSTEM IMPORTS
import numpy
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_cd_, "..")
for _dir_ in [_cd_, _src_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import basenet
import core


class ANN(basenet.BaseNet):
    def __init__(self, layers, seed=None, afuncs=None, afunc_primes=None,
                 learning_rate=1.0, weight_decay_coeff=0.0, ignore_overflow=False):
        super(ANN, self).__init__(layers, seed=seed, afuncs=afuncs,
                                  afunc_primes=afunc_primes, ignore_overflow=ignore_overflow)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay_coeff

    def cost_function(self, X, Y):
        y_hat = self.feed_forward(X)
        cost = 0.5 * numpy.sum((y_hat - Y) ** 2) / X.shape[0]  # normalization constant

        weight_decay_term = 0.0
        if self.weight_decay != 0.0:
            weight_decay_term = (self.weight_decay / 2.0) *\
                (numpy.sum(numpy.sum(self.weights ** 2, axis=1)))
                # (numpy.sum([numpy.sum(w ** 2) for w in self.weights])) # + sum([sum(b ** 2) for b in self._biases]))
        return cost + weight_decay_term

    def back_propogate(self, X, Y):
        old_settings = dict()
        if not self.ignore_overflow:
            old_settings = numpy.seterr(over="raise")
        else:
            old_settings = numpy.seterr(over="ignore")

        try:
            # feed forward but remember each layer's computations as we go
            zs = list()
            as_ = list([X])
            a = X
            z = None
            for afunc, weight, bias in zip(self.afuncs, self.weights, self.biases):
                z = numpy.dot(a, weight) + bias
                zs.append(z)
                a = afunc(z)
                as_.append(a)

            dLdWs = [None for _ in self.weights]
            dLdBs = [None for _ in self.biases]

            # compute the last layer first
            delta = numpy.multiply((as_[-1] - Y), self.afunc_primes[-1](zs[-1]))
            dLdBs[-1] = numpy.sum(delta, axis=0, keepdims=True) + self.weight_decay*self.biases[-1]
            dLdWs[-1] = numpy.dot(as_[-2].T, delta) + self.weight_decay*self.weights[-1]

            for i in range(2, len(self.weights) + 1):
                delta = numpy.dot(delta, self.weights[-i+1].T) * self.afunc_primes[-i](zs[-i])
                dLdBs[-i] = numpy.sum(delta, axis=0) + self.weight_decay*self.biases[-i]
                dLdWs[-i] = numpy.dot(as_[-i-1].T, delta) + self.weight_decay*self.weights[-i]
            return dLdWs, dLdBs
        except FloatingPointError:
            raise FloatingPointError("Overflow occured, please scale features")
        finally:
            numpy.seterr(**old_settings)

    def _train(self, X, Y):
        dLdWs, dLdBs = self.back_propogate(X, Y)
        self.weights = [w - self.learning_rate * dLdW for w, dLdW in zip(self.weights, dLdWs)]
        self.biases = [b - self.learning_rate * dLdB for b, dLdB in zip(self.biases, dLdBs)]

