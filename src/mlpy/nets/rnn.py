# SYSTEM IMPORTS
import numpy
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
# _src_dir_ = os.path.join(_cd_, "..")
for _dir_ in [_cd_]: # , _src_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
# del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import activations as af
import basernn


class rnn(basernn.BaseRNN):
    def __init__(self, input_size, output_size, hidden_size=100, seed=None, afuncs=None, afunc_primes=None, bptt_truncate=4, learning_rate=0.005, loss_func=None):
        super(rnn, self).__init__(input_size, output_size, afuncs=afuncs, seed=seed,
                                  afunc_primes=afunc_primes, bptt_truncate=bptt_truncate,
                                  loss_func=loss_func)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.W = numpy.random.uniform(-1.0, 1.0, size=(hidden_size, hidden_size,))
        self.U = numpy.random.uniform(-1.0, 1.0, size=(hidden_size, input_size,))
        self.V = numpy.random.uniform(-1.0, 1.0, size=(output_size, hidden_size,))
        self.S = numpy.zeros(hidden_size)

        self.b_s = numpy.random.uniform(-1.0, 1.0, size=(hidden_size))
        self.b_o = numpy.random.uniform(-1.0, 1.0, size=(output_size))

        if self.afuncs is None:
            self.afuncs = [af.tanh, af.softmax]
        if self.afunc_primes is None:
            self.afunc_primes = [af.tanh_prime, af.softmax_prime]

    def compute_layer(self, X):
        X = self._assert_numpy(X)
        self.S = self.afuncs[0](numpy.dot(self.U, X) + numpy.dot(self.W, self.S) + self.b_s)
        return self.afuncs[1](numpy.dot(self.V, self.S) + self.b_o)

    def compute_layer_and_cache(self, X):
        X = self._assert_numpy(X)
        S_args = numpy.dot(self.U, X) + numpy.dot(self.W, self.S) + self.b_s
        self.S = self.afuncs[0](S_args)
        O_args = numpy.dot(self.V, self.S) + self.b_o
        O = self.afuncs[1](O_args)
        return (S_args, numpy.array(self.S), O_args, O)

    def reset(self):
        self.S = numpy.zeros(self.S.shape)

    def back_propagate_through_time(self, X, Y):
        assert(X.shape[1] == self.input_size)
        X = self._assert_numpy(X)
        Y = self._assert_numpy(Y)
        self.reset()
        S_args, Ss, O_args, Os = self.feed_forward_and_cache(X)
        Ss = numpy.concatenate((Ss, numpy.zeros((1, self.hidden_size))))
        self.reset()

        dLdU = numpy.zeros(self.U.shape)
        dLdV = numpy.zeros(self.V.shape)
        dLdW = numpy.zeros(self.W.shape)

        dLdb_s = numpy.zeros(self.b_s.shape)
        dLdb_o = numpy.zeros(self.b_o.shape)

        dLdX = numpy.zeros(X.shape)

        delta = self.compute_error_vector(Os, Y)
        for t in range(X.shape[0]):  # walk backwards through time
            delta_t = numpy.multiply(delta[t], self.afunc_primes[1](O_args[t]))
            dLdV += numpy.outer(delta_t, Ss[t])
            dLdb_o += delta_t
            delta_t = numpy.multiply(numpy.dot(delta_t, self.V),
                                     self.afunc_primes[0](S_args[t]))
            for bptt_step in range(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdb_s += delta_t
                dLdW += numpy.outer(delta_t, Ss[bptt_step-1])
                dLdU += numpy.outer(delta_t, X[bptt_step])
                dLdX[bptt_step] += numpy.dot(delta_t, self.U)
                delta_t = numpy.multiply(numpy.dot(delta_t, self.W),
                                         self.afunc_primes[0](Ss[bptt_step-1]))
                # print(delta_t.shape)
        return (dLdU, dLdV, dLdW, dLdX,)

    def _train_return_errors(self, X, Y):
        self.reset()
        dLdU, dLdV, dLdW, dLdX = self.back_propagate_through_time(X, Y)
        self.U -= self.learning_rate*dLdU
        self.V -= self.learning_rate*dLdV
        self.W -= self.learning_rate*dLdW
        self.reset()
        return dLdX

    """
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
    """

