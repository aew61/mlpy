# SYSTEM IMPORTS
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
import basenet
import core


class cbow(basenet.BaseNet):
    def __init__(self, vocab_size, context_size, num_embedding_dims):
        super([vocab_size, num_embedding_dims, vocab_size],
              afuncs=[af.linear, af.softmax],
              afunc_primes=[af.linear_prime, af.softmax_prime])
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.biases = None

    def avg_input_to_hidden_weights(self, W):
        return numpy.mean([a[numpy.arange(a.shape[0]), i*self.vocab_size:(i+1)*self.vocab_size]
                          for i in range(self.context_size)], axis=0)

    def cost_function(self, X, y):
        return y - self.feed_forward(X)

    def feed_forward(self, X):
        new_settings = dict({"over": "ignore"})
        if not self.ignore_overflow:
            new_settings["over"] = "raise"
        old_settings = self.change_settings(new_settings)

        try:
            a = X
            for i, (afunc, weight) in enumerate(zip(self.afuncs, self.weights)):
                a = afunc(numpy.dot(a, weight))
                if i == 0:
                    # need to average all values.
                    # a has size (num_examples, context_size * vocab_size)
                    # split into "context_size" arrays of size (num_examples, vocab_size)
                    # and then compute the average of them
                    a = self.avg_input_to_hidden_weights(a)
            return a
        except FloatingPointError:
            raise FloatingPointError("Overflow occured, please scale features")
        finally:
            self.change_settings(old_settings)

    def back_propagate(self, X, Y):
        new_settings = dict({"over": "ignore"})
        if not self.ignore_overflow:
            new_settings["over"] = "raise"
        old_settings = self.change_settings(new_settings)

        try:
            # feed forward but remember each layer's computations as we go
            zs = list()
            as_ = list([X])
            a = X
            z = None
            for i, (afunc, weight) in enumerate(zip(self.afuncs, self.weights)):
                z = numpy.dot(a, weight) + bias
                zs.append(z)
                a = afunc(z)
                if i == 0:
                    a = self.avg_input_to_hidden_weights(a)
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
            self.change_settings(old_settings)

    def _train(self, X, Y):
        dLdWs = self.back_propagate(X, Y)
        self.weights = [w - self.learning_rate * dLdW for w, dLdW in zip(self.weights, dLdWs)]

    def _predict_example(self, x):
        return self.weights[-1].T[x == 1]

