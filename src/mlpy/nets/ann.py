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
from activation_functions import sigmoid, sigmoid_prime
import basenet
import core


class ANN(basenet.BaseNet):
    def __init__(self, layer_sizes, seed=None, afunc_ptrs=None, ignore_biases=False,
                 learning_rate=1.0, weight_decay_coeff=0.0, ignore_overflow=False):
        super(ANN, self).__init__(layer_sizes, seed=seed, afunc_ptrs=afunc_ptrs,
                                  ignore_biases=ignore_biases, ignore_overflow=ignore_overflow)

        self.afunc_prime_ptr = None
        if self.afunc_ptr == sigmoid:
            self.afunc_prime_ptr = sigmoid_prime
        else:
            self.afunc_prime_ptr = afunc_ptrs[1]

        self.learning_rate = learning_rate
        self.weight_decay_coeff = weight_decay_coeff

    def cost_function(self, X, Y):
        y_hat = self.feed_forward(X)
        cost = 0.5 * numpy.sum((y_hat - Y) ** 2) / X.shape[0]  # normalization constant

        weight_decay_term = 0.0
        if self.weight_decay_coeff != 0.0:
            weight_decay_term = (self.weight_decay_coeff / 2.0) *\
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
            ns = list()
            activations = list([X])
            a = X
            n = None
            for weight, bias in zip(self.weights, self.biases):
                n = numpy.dot(a, weight)
                ns.append(n)
                a = self.afunc_ptr(n)
                activations.append(a)

            # print(activations[-1].shape)

            dLdWs = [numpy.zeros(w.shape) for w in self.weights]
            dLdBs = [numpy.zeros(b.shape) for b in self.biases]

            # compute the last layer first
            # print((activations[-1] - Y).shape)
            # print(self.afunc_prime_ptr(ns[-1]).shape)
            delta = numpy.multiply((activations[-1] - Y), self.afunc_prime_ptr(ns[-1]))
            dLdWs[-1] = numpy.dot(activations[-2].T, delta)
            # print(dLdWs[-1].shape)

            if self.weight_decay_coeff != 0.0:
                dLdWs[-1] += weight_decay_coeff * self.weights[-1]
            for index in range(2, len(self.weights) + 1):
                delta = numpy.dot(delta, self.weights[-index + 1].T) * self.afunc_prime_ptr(ns[-index])
                dLdWs[-index] = numpy.dot(activations[-index - 1].T, delta)
                # print(dLdWs[-index].shape)
                if self.weight_decay_coeff != 0.0:
                    dLdWs[-index] += self.weight_decay_coeff * self.weights[-index]
            return dLdWs, dLdBs
        except FloatingPointError:
            raise FloatingPointError("Overflow occured, please scale features")
        finally:
            numpy.seterr(**old_settings)

    def _train(self, X, Y):
        dLdWs, dLdBs = self.back_propogate(X, Y)
        self.weights = [w - self.learning_rate * dLdW for w, dLdW in zip(self.weights, dLdWs)]

