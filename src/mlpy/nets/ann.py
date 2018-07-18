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


class ann(basenet.BaseNet):
    def __init__(self, layers, seed=None, afuncs=None, afunc_primes=None,
                 learning_rate=1.0, weight_decay_coeff=0.0, ignore_overflow=False,
                 loss_func=None): #, error_func=None):
        super(ann, self).__init__(layers, seed=seed, afuncs=afuncs, loss_func=loss_func,
                                  afunc_primes=afunc_primes, ignore_overflow=ignore_overflow) #,
                                  # error_func=error_func)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay_coeff

    def loss_function(self, X, Y):
        cost = super(ann, self).loss_function(X, Y)

        weight_decay_term = 0.0
        if self.weight_decay != 0.0:
            weight_decay_term = (self.weight_decay / 2.0) *\
                (numpy.sum(numpy.sum(self.weights ** 2, axis=1)))
                # (numpy.sum([numpy.sum(w ** 2) for w in self.weights])) # + sum([sum(b ** 2) for b in self._biases]))
        return cost + weight_decay_term

    def complete_feed_forward(self, X):
        zs = list()
        as_ = list([X])
        a = X
        z = None
        for afunc, weight, bias in zip(self.afuncs, self.weights, self.biases):
            z = numpy.dot(a, weight) + bias
            zs.append(z)
            a = afunc(z)
            as_.append(a)
        return zs, as_

    def compute_error_vector(self, Y_hat, Y):
        return Y_hat - Y

    def back_propagate(self, X, Y):
        new_settings = dict({"over": "ignore"})
        if not self.ignore_overflow:
            new_settings["over"] = "raise"
        old_settings = self.change_settings(new_settings)

        try:
            # feed forward but remember each layer's computations as we go
            zs, as_ = self.complete_feed_forward(X)

            dLdWs = [None for _ in self.weights]
            dLdBs = [None for _ in self.biases]

            # compute the last layer first
            delta = self.compute_error_vector(as_[-1], Y)
            afunc_prime = self.afunc_primes[-1](zs[-1])
            if len(afunc_prime.shape) == 3:
                delta = numpy.einsum("...jk, ...kl", afunc_prime,
                                     delta.reshape(delta.shape+(1,)), optimize=True).reshape(delta.shape)
            else:
                delta = numpy.multiply(delta, afunc_prime)

            # delta = numpy.multiply(self.compute_error_vector(as_[-1], Y), self.afunc_primes[-1](zs[-1]))
            dLdBs[-1] = numpy.sum(delta, axis=0, keepdims=True) + self.weight_decay*self.biases[-1]
            dLdWs[-1] = numpy.dot(as_[-2].T, delta) + self.weight_decay*self.weights[-1]

            for i in range(2, len(self.weights) + 1):
                delta = numpy.dot(delta, self.weights[-i+1].T)
                afunc_prime = self.afunc_primes[-i](zs[-i])
                if len(afunc_prime.shape) == 3:
                    delta = numpy.einsum("...jk, ...kl", afunc_prime,
                                         delta.reshape(delta.shape+(1,)), optimize=True).reshape(delta.shape)
                else:
                    delta = numpy.multiply(delta, afunc_prime)

                # delta = numpy.dot(delta, self.weights[-i+1].T) * self.afunc_primes[-i](zs[-i])
                dLdBs[-i] = numpy.sum(delta, axis=0) + self.weight_decay*self.biases[-i]
                dLdWs[-i] = numpy.dot(as_[-i-1].T, delta) + self.weight_decay*self.weights[-i]
            dLdX = numpy.dot(delta, self.weights[0].T)
            return dLdWs, dLdBs, dLdX
        except FloatingPointError:
            raise FloatingPointError("Overflow occured, please scale features")
        finally:
            self.change_settings(old_settings)

    def _train_return_errors(self, X, Y):
        dLdWs, dLdBs, dLdX = self.back_propagate(X, Y)
        self.weights = [w - self.learning_rate * dLdW for w, dLdW in zip(self.weights, dLdWs)]
        self.biases = [b - self.learning_rate * dLdB for b, dLdB in zip(self.biases, dLdBs)]
        return dLdX

