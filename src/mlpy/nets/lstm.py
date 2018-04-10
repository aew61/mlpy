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
import basernn


class lstm(basernn.BaseRNN):
    def __init__(self, input_size, output_size, hidden_size=100, seed=None, afuncs=None,
                 afunc_primes=None, bptt_truncate=4, learning_rate=0.005):
        super(lstm, self).__init__(input_size, output_size, afuncs=afuncs, seed=seed,
                                   afunc_primes=afunc_primes, bptt_truncate=bptt_truncate)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.S = numpy.zeros(self.hidden_size)
        self.h_t = numpy.zeros(self.hidden_size)

        W_shape = (self.hidden_size, self.input_size)
        U_shape = (self.hidden_size, self.hidden_size)
        b_shape = (self.hidden_size)

        self.W_f = numpy.random.uniform(-1.0, 1.0, size=W_shape)
        self.U_f = numpy.random.uniform(-1.0, 1.0, size=U_shape)
        self.b_f = numpy.ones(b_shape)

        self.W_i = numpy.random.uniform(-1.0, 1.0, size=W_shape)
        self.U_i = numpy.random.uniform(-1.0, 1.0, size=U_shape)
        self.b_i = numpy.random.uniform(-1.0, 1.0, size=b_shape)

        self.W_c = numpy.random.uniform(-1.0, 1.0, size=W_shape)
        self.U_c = numpy.random.uniform(-1.0, 1.0, size=U_shape)
        self.b_c = numpy.random.uniform(-1.0, 1.0, size=b_shape)

        self.W_hf = numpy.random.uniform(-1.0, 1.0, size=W_shape)
        self.U_hf = numpy.random.uniform(-1.0, 1.0, size=U_shape)
        self.b_hf = numpy.random.uniform(-1.0, 1.0, size=b_shape)

        self.W_o = numpy.random.uniform(-1.0, 1.0, size=(self.output_size, self.hidden_size))
        self.b_o = numpy.random.uniform(-1.0, 1.0, size=(self.output_size))

        if self.afuncs is None:
            self.afuncs = [af.tanh, af.tanh, af.softmax]
        if self.afunc_primes is None:
            self.afunc_primes = [af.tanh_prime, af.tanh_prime, af.softmax_prime]

    def compute_layer(self, X):
        # print(self.h_t.shape)
        # print(X.shape)
        # compute the "forget gate"
        F = af.sigmoid(numpy.dot(self.W_f, X) + numpy.dot(self.U_f, self.h_t) + self.b_f)
        # compute the "input gate"
        I = af.sigmoid(numpy.dot(self.W_i, X) + numpy.dot(self.U_i, self.h_t) + self.b_i)
        # compute "candidate values"
        C = self.afuncs[0](numpy.dot(self.W_c, X) + numpy.dot(self.U_c, self.h_t) + self.b_c)
        # update the state
        self.S = numpy.multiply(F, self.S) + numpy.multiply(I, C)
        # compute the "output filter"
        Hf = af.sigmoid(numpy.dot(self.W_hf, X) + numpy.dot(self.U_hf, self.h_t) + self.b_hf)
        # compute the final "hidden state"
        self.h_t = numpy.multiply(Hf, self.afuncs[1](self.S))
        # convert the final hidden state to the observed state
        return self.afuncs[2](numpy.dot(self.W_o, self.h_t) + self.b_o)

    def reset(self):
        self.S[:] = 0
        self.h_t[:] = 0

    def back_propagate_through_time(self, X, Y):
        assert(X.shape[1] == self.input_size)
        time = X.shape[0]
        F_args = numpy.zeros((time, self.hidden_size))
        Fs = numpy.zeros(F_args.shape)

        I_args = numpy.zeros((time, self.hidden_size))
        Is = numpy.zeros(I_args.shape)

        C_args = numpy.zeros((time, self.hidden_size))
        Cs = numpy.zeros(C_args.shape)

        Hf_args = numpy.zeros((time, self.hidden))
        Hfs = numpy.zeros(Hf_args.shape)

        Hs = numpy.zeros((time, self.hidden_size))
        Hs[-1] = numpy.zeros(self.hidden_size)

        O_args = numpy.zeros((time, self.output_size))
        Os = numpy.zeros(O_args.shape)

        Ss = numpy.zeros((time+1, self.hidden_size))
        Ss[-1] = numpy.zeros(self.hidden_size)
        for t in range(time):
            # forward propagate, saving information as we go
            F_args[t] = numpy.dot(self.W_f, X) + numpy.dot(self.U_f, self.h_t) + self.b_f
            Fs[t] = af.sigmoid(F_args[t])

            I_args[t] = numpy.dot(self.W_i, X) + numpy.dot(self.U_i, self.h_t) + self.b_i
            Is[t] = af.sigmoid(I_args[t])

            C_args[t] = numpy.dot(self.W_c, X) + numpy.dot(self.U_c, self.h_t) + self.b_c
            Cs[t] = self.afuncs[0](C_args[t])

            Ss[t] = numpy.multiply(F, Ss[t-1]) + numpy.multiply(I, C)

            Hf_args[t] = numpy.dot(self.W_hf, X) + numpy.dot(self.U_hf, self.h_t) + self.b_hf
            Hfs[t] = af.sigmoid(H_args[t])

            Hs[t] = numpy.multiply(Hfs[t], self.afuncs[1](Ss[t]))

            O_args[t] = numpy.dot(self.W_o, Hs[t]) + self.b_o
            Os[t] = self.afuncs[2](O_args[t])

        dLdW_f = numpy.zeros(self.W_f.shape)
        dLdU_f = numpy.zeros(self.U_f.shape)
        dLdb_f = numpy.zeros(self.b_f.shape)

        dLdW_i = numpy.zeros(self.W_i.shape)
        dLdU_i = numpy.zeros(self.U_i.shape)
        dLdb_i = numpy.zeros(self.b_i.shape)

        dLdW_c = numpy.zeros(self.W_c.shape)
        dLdU_c = numpy.zeros(self.U_c.shape)
        dLdb_c = numpy.zeros(self.b_c.shape)

        dLdW_hf = numpy.zeros(self.W_hf.shape)
        dLdU_hf = numpy.zeros(self.U_hf.shape)
        dLdb_hf = numpy.zeros(self.b_hf.shape)

        # (Os - Y) has dimensions (num_examples, output_size)
        # and O_args has dimensions (num_examples, output_size)
        # so delta has dimensions (num_examples, output_size)
        delta = numpy.multiply(Os - Y, self.afunc_primes[2](O_args))
        dLdb_o = numpy.sum(delta, axis=0)

        # dLdW_o is more complex. The derivative for a single element is
        # h_t * delta, however, h_t has dimension (hidden_size) and
        # delta_h has dimensions (num_examples, output_size)
        # Hs on the other hand has dimensions (num_examples, hidden_size)
        # and dLdW_o should have dimensions (output_size, hidden_size)
        # so dLdW_o, to make dimensions agree, should be delta.T * Hs
        dLdW_o = numpy.dot(delta.T, Hs)
        for t in range(time):
            # to compute the error due to ht, element-wise, the equation is:
            # delta_h = delta * W_o.
            # A row of delta has dimensions (output_size) and W_o has
            # dimensions (output_size, hidden_size), so a row of delta_h must have dimensions
            # (hidden_size). This means delta_h = dot(delta, W_o)
            delta_h = numpy.dot(delta, W_o)
            # delta_s_t will be the error that the "top data bus" that holds
            # s_(t-1) -> s_t transformation on it.
            # Likewise, delta_h_t will be the error that the "bottom data bus"
            # that holds the [h_(t-1), x_t] -> h_t transformation on it.
            # We need to compute this value before the back-through-time part
            # because initially, this error is only coming from o_t, while when
            # we start to go back through time, error along both busses will compound

            # We know h_t = h_f * f_2(s_t), so delta_s_t, initially should be
            # delta_h[t] * h_f * f'_2(s_t), each variable having dimensions (hidden_size)
            delta_s_t = numpy.multiply(numpy.multiply(delta_h, Hfs[t]),
                                       self.afunc_primes[1](Ss[t]))
            # differentiating h_t w/ respect to h_f, we have:
            # delta_h[t] * f_2(s_t)
            delta_h_t = numpy.multiply(delta_h, self.afuncs[1](Ss[t]))
 
            for bptt_step in range(max(0, t-self.bptt_truncate), t+1)[::-1]:
                pass
                # need to update delta_t_s and delta_t_o
                # delta_s_t = numpy.multiply(delta_s_t, delta_s_t_minus_one)
                # delta_h_t += delta_h_f + delta_h_i + delta_h_c + delta_h_hf

        return (dLdW_f, dLdU_f, dLdb_f), (dLdW_i, dLdU_i, dLdb_i),\
            (dLdW_c, dLdU_c, dLdb_c), (dLdW_hf, dLdU_hf, dLdb_hf), (dLdW_o, dLdb_o)

    def _train(self, X, Y):
        for i in range(len(Y)):
            self.reset()
            dLdF, dLdI, dLdC, dLdHf, dLdO = self.back_propagate_through_time(X[i], Y[i])

            self.W_f -= self.learning_rate*dLdF[0]
            self.U_f -= self.learning_rate*dLdF[1]
            self.b_f -= self.learning_rate*dLdF[2]

            self.W_i -= self.learning_rate*dLdI[0]
            self.U_i -= self.learning_rate*dLdI[1]
            self.b_i -= self.learning_rate*dLdI[2]

            self.W_c -= self.learning_rate*dLdC[0]
            self.U_c -= self.learning_rate*dLdC[1]
            self.b_c -= self.learning_rate*dLdC[2]

            self.W_hf -= self.learning_rate*dLdHf[0]
            self.U_hf -= self.learning_rate*dLdHf[1]
            self.b_hf -= self.learning_rate*dLdHf[2]

            self.W_o -= self.learning_rate*dLdO[0]
            self.b_o -= self.learning_rate*dLdO[1]
        self.reset()
        return self

