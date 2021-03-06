# SYSTEM IMPORTS
from abc import ABCMeta, abstractmethod
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
from activations import softmax, softmax_prime
import core
from losses import cross_entropy, squared_difference


class BaseRNN(core.Base, metaclass=ABCMeta):
    def __init__(self, input_size, output_size, bptt_truncate=4, afuncs=None, afunc_primes=None, seed=None, loss_func=None):
        super(BaseRNN, self).__init__()
        self.afuncs = afuncs
        self.afunc_primes = afunc_primes
        numpy.random.seed(seed)
        self.input_size = input_size
        self.output_size = output_size
        self.bptt_truncate = bptt_truncate
        self.loss_func = loss_func
        if self.loss_func is None:
            if self.afuncs is None or (self.afuncs[-1] == softmax and self.afunc_primes[-1] == softmax_prime):
                self.loss_func = cross_entropy
            else:
                self.loss_func = squared_difference

    @abstractmethod
    def compute_layer(self, X):
        pass

    @abstractmethod
    def compute_layer_and_cache(self, X):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def back_propagate_through_time(self, X, Y):
        pass

    @abstractmethod
    def _train_return_errors(self, X, Y):
        pass

    def compute_error_vector(self, Y_hat, Y):
        return Y_hat - Y

    def _assert_numpy(self, X):
        if not isinstance(X, numpy.ndarray):
            return X.toarray()
        return X

    def _train(self, X, Y, verbose=0, epochs=1, converge_function=None):
        print_increment = 0.001
        for epoch in range(epochs):
            tot = len(Y)
            current = 0
            current_loss = 0
            N = 0
            print_threshold = 0.0

            for i in range(len(Y)):
                self.reset()
                self._train_return_errors(X[i], Y[i])
                self.reset()

                if verbose > 0:
                    current += 1
                    N += X[i].shape[0]
                    current_loss += self.loss_function([X[i]], [Y[i]])*X[i].shape[0]

                    if verbose > 1 and float(current)/tot > print_threshold:
                        print_threshold += print_increment
                        print("training epoch {0}/{1} [{2:.1f}%] complete | loss [{3:.3f}]\
                              \r".format((epoch+1), epochs, float(current)*100/tot,
                                         current_loss/N), end="", flush=True)
                    elif float(current)/tot > print_threshold:
                        print_threshold += print_increment
                        print("training epoch {0}/{1} [{2:.1f}%] complete\r".format(
                              (epoch+1), epochs, float(current)*100/tot), end="", flush=True)

            if verbose > 1:
                print("training epoch {0}/{1} [{2:.1f}%] complete | loss [{3:.3f}]\r"
                      .format((epoch+1), epochs, float(current)*100/tot, current_loss/N))
            elif verbose > 0:
                 print("training epoch {0}/{1} [{2:.1f}%] complete\r".format(
                       (epoch+1), epochs, float(current)*100/tot))

            if converge_function is not None and converge_function(self):
                return self

        return self

    def _predict_example(self, x):
        return self.predict(x.reshape(1, x.shape[0]))

    def feed_forward(self, X):
        X = self._assert_numpy(X)
        Os = numpy.zeros((X.shape[0], self.output_size))
        for i in range(X.shape[0]):  # for each example
            Os[i] = self.compute_layer(X[i])
        return Os

    def feed_forward_and_cache(self, X):
        X = self._assert_numpy(X)
        cache_list = list()
        for i in range(X.shape[0]):
            cache_list.append(self.compute_layer_and_cache(X[i]))
        return [numpy.array([cache_list[a][b] for a in range(len(cache_list))]) for b in range(len(cache_list[0]))]

    def predict(self, X):
        assert(X.shape[1] == self.input_size)
        Os = self.feed_forward(X)
        max_indices = numpy.argmax(Os, axis=1)
        Os[:, :] = 0
        Os[range(Os.shape[0]), max_indices] = 1
        return Os

    def loss_function(self, X, Y):
        L = 0
        N = 0
        num_examples = len(X)
        for i in range(num_examples):
            self.reset()
            N += X[i].shape[0]
            Os = self.predict_proba(X[i])
            # L += -1*numpy.sum(numpy.log(Os[range(X[i].shape[0]), numpy.argmax(Y[i], axis=1)]))
            L += X[i].shape[0] * self.loss_func(Os, Y[i])
        self.reset()
        return L/N

    def predict_proba(self, X):
        assert(X.shape[1] == self.input_size)
        return self.feed_forward(X)

