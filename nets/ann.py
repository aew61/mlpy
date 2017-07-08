# SYSTEM IMPORTS
import numpy
import random


# PYTHON PROJECT IMPORTS
from activation_functions import sigmoid, sigmoid_prime


class ann(object):
    def __init__(self, layers, active_func_ptr=None, active_func_prime_ptr=None):
        random.seed(12345)

        # parse out negative or 0 layer specifications
        filtered_layers = [layer for layer in layers if layer > 0]
        self._num_layers = len(filtered_layers)
        self._weights = [numpy.array([random.uniform(-1.0, 1.0)
                                      for z in range(row * col)]).reshape(row, col)
                         for row, col in zip(filtered_layers[:-1], filtered_layers[1:])]

        self._biases = [numpy.array([random.uniform(-1.0, 1.0)
                                     for z in range(n)]).reshape(1, n)
                        for n in filtered_layers[1:]]

        self._active_func_ptr = active_func_ptr
        self._active_func_prime_ptr = active_func_prime_ptr
        if self._active_func_ptr is None or self._active_func_prime_ptr is None:
            self._active_func_ptr = sigmoid
            self._active_func_prime_ptr = sigmoid_prime

    def cost_function(self, X, Y, weight_decay_coeff=0.0):
        y_hat = self.feed_forward(X)
        cost = 0.5 * sum((y_hat - Y) ** 2) / X.shape[0]  # normalization constant

        weight_decay_term = 0.0
        if weight_decay_coeff != 0.0:
            # for weight in self._weights:
            #     weight_decay_term += sum(weight ** 2)
            # weight_decay_term *= weight_decay_coeff
            weight_decay_term = (weight_decay_coeff / 2.0) *\
                (sum([sum(w ** 2) for w in self._weights])) # + sum([sum(b ** 2) for b in self._biases]))
        return cost + weight_decay_term

    def feed_forward(self, X):
        a = X
        for weight, bias in zip(self._weights, self._biases):
            # print("weight: %s" % weight)
            # print("bias: %s" % bias)
            # print("shape of numpy.dot(a, weight): %s, bias shape:%s" % (weight.shape, bias.shape))
            a = self._active_func_ptr(numpy.dot(a, weight)) # + bias)
        return a

    def back_propogate(self, X, Y, weight_decay_coeff=0.0):
        # feed forward but remember each layer's computations as we go
        ns = list()
        activations = list([X])
        a = X
        n = None
        for weight, bias in zip(self._weights, self._biases):
            n = numpy.dot(a, weight)
            ns.append(n)
            a = self._active_func_ptr(n)
            activations.append(a)

        dLdWs = [numpy.zeros(w.shape) for w in self._weights]
        dLdBs = [numpy.zeros(b.shape) for b in self._biases]

        # compute the last layer first
        delta = numpy.multiply((activations[-1] - Y), self._active_func_prime_ptr(ns[-1]))
        # print("shape of dot(activations[-2].T, delta): %s, shape of weights[-1]: %s" %
        #       (numpy.dot(activations[-2].T, delta).shape, self._weights[-1].shape))
        dLdWs[-1] = numpy.dot(activations[-2].T, delta)
        if weight_decay_coeff != 0.0:
            dLdWs[-1] += weight_decay_coeff * self._weights[-1]
        for index in range(2, len(self._weights) + 1):
            delta = numpy.dot(delta, self._weights[-index + 1].T) * self._active_func_prime_ptr(ns[-index])
            dLdWs[-index] = numpy.dot(activations[-index - 1].T, delta)
            if weight_decay_coeff != 0.0:
                dLdWs[-index] += weight_decay_coeff * self._weights[-index]
        return dLdWs, dLdBs

    def train_on_data_set(self, X, Y, learning_rate=1.0, weight_decay_coeff=0.0):
        dLdWs, dLdBs = self.back_propogate(X, Y, weight_decay_coeff=weight_decay_coeff)
        # print("dLdWs:\n%s" % dLdWs)
        self._weights = [w - learning_rate * dLdW for w, dLdW in zip(self._weights, dLdWs)]
        # self._biases = [b - learning_rate * dLdB for b, dLdB in zip(self._biases, dLdBs)]

