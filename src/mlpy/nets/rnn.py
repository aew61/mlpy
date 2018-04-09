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
import activation_functions as af


class rnn(object):
    def __init__(self, input_size, output_size, hidden_size=100, seed=None, afuncs=None, afunc_primes=None, bptt_truncate=4, learning_rate=0.005):
        numpy.random.seed(seed)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bptt_truncate = bptt_truncate
        self.learning_rate = learning_rate

        self.W = numpy.random.uniform(-1.0, 1.0, size=(hidden_size, hidden_size,))
        self.U = numpy.random.uniform(-1.0, 1.0, size=(hidden_size, input_size,))
        self.V = numpy.random.uniform(-1.0, 1.0, size=(output_size, hidden_size,))
        self.S = numpy.zeros(hidden_size)

        self.afuncs = afuncs
        self.afunc_primes = afunc_primes

        if self.afuncs is None:
            self.afuncs = [af.tanh, af.softmax]
        if self.afunc_primes is None:
            self.afunc_primes = [af.tanh_prime, af.softmax_prime]

    def feed_forward(self, X):
        time = X.shape[0]
        Os = numpy.zeros((time, self.output_size))
        for t in range(time):
            self.S = self.afuncs[0](numpy.dot(self.U, X[t]) + numpy.dot(self.W, self.S))
            Os[t] = self.afuncs[1](numpy.dot(self.V, self.S))
        return Os

    def predict(self, X):
        assert(X.shape[1] == self.input_size)
        return self.feed_forward(X)

    def loss_function(self, X, Y):
        L = 0
        N = 0
        num_examples = len(X)
        cached_s = self.S
        for i in range(num_examples):
            N += X[i].shape[0]
            Os = self.predict(X[i])
            self.S = cached_s
            correct_predictions = Os[range(X[i].shape[0]), numpy.argmax(Y[i], axis=1)]
            L += -1*numpy.sum(numpy.log(correct_predictions))
        return L/N

    def back_propagate_through_time(self, X, Y):
        assert(X.shape[1] == self.input_size)
        time = X.shape[0]
        Os = numpy.zeros((time, self.output_size))
        Ss = numpy.zeros((time+1, self.hidden_size))
        for t in range(time):
            Ss[t+1] = self.afuncs[0](numpy.dot(self.U, X[t]) + numpy.dot(self.W, Ss[t]))
            Os[t] = self.afuncs[1](numpy.dot(self.V, Ss[t+1]))

        dLdU = numpy.zeros(self.U.shape)
        dLdV = numpy.zeros(self.V.shape)
        dLdW = numpy.zeros(self.W.shape)

        delta = Os - Y
        # print(delta.shape)
        for t in range(time):  # walk backwards through time
            dLdV += numpy.multiply(numpy.outer(delta[t].T, Ss[t+1]), self.afunc_primes[1](Ss[t]))
            # dLdV += numpy.multiply(delta[t], self.afunc_primes[1](Ss[t]))
            # print(delta[t].shape)
            # print(self.V.T.shape)
            # print((Ss[t+1] ** 2).shape)
            delta_t = numpy.multiply(numpy.dot(self.V.T, delta[t]),
                                     self.afunc_primes[0](Ss[t+1]))
            # delta_t = self.V.T.dot(delta[t]) * (1 - (Ss[t+1] ** 2))
            # print("delta_t.shape")
            for bptt_step in range(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print(delta_t)
                dLdW += numpy.outer(delta_t, Ss[bptt_step])
                dLdU[:, numpy.argmax(X[bptt_step])] += delta_t
                delta_t = numpy.multiply(numpy.dot(self.W.T, delta_t),
                                         self.afunc_primes[0](Ss[bptt_step-2]))
                # print(delta_t.shape)
        return [dLdU, dLdV, dLdW]

    def train(self, X, Y):
        for i in range(len(Y)):
            dLdU, dLdV, dLdW = self.back_propagate_through_time(X[i], Y[i])
            self.U -= self.learning_rate*dLdU
            self.V -= self.learning_rate*dLdV
            self.W -= self.learning_rate*dLdW
        return self

