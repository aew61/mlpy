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


class lstm(basernn.BaseRNN):
    def __init__(self, input_size, output_size, hidden_size=100, seed=None, afuncs=None,
                 afunc_primes=None, bptt_truncate=4, learning_rate=0.005, loss_func=None):
        super(lstm, self).__init__(input_size, output_size, afuncs=afuncs, seed=seed,
                                   afunc_primes=afunc_primes, bptt_truncate=bptt_truncate,
                                   loss_func=loss_func)
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

    def back_prop_forget_gate(self, delta_s_t, delta_h_t, S, F_arg, H, X, dLdfs):
        dLdW_f, dLdU_f, dLdb_f = dLdfs

        # last gate: The "forget" gate. We know s_t = F * s_t-1 + I * C
        # so delta_f = delta_s_t * s_t-1, both element have size (hidden_size)
        delta_f = numpy.multiply(delta_s_t, S)
        # we also know F = sigmoid(dot(W_f, x_t) + dot(U_f, h_t-1) + b_f)
        # we can cache some of this information.
        delta_f = numpy.multiply(delta_f, af.sigmoid_prime(F_arg))
        dLdb_f += delta_f
        dLdU_f += numpy.outer(delta_f, H)
        dLdW_f += numpy.outer(delta_f, X)
        # need to update delta_h_t: each element += delta_f * U_f.T
        delta_h_t += numpy.dot(delta_f, self.U_f.T)

    def back_prop_input_gate(self, delta_s_t, delta_h_t, C, I_arg, H, X, dLdis):
        dLdW_i, dLdU_i, dLdb_i = dLdis

        # The "input gate." We know s_t = F * s_t-1 + I * C
        # so delta_i = delta_s_t * C. Both parts have shape (hidden_state)
        delta_i = numpy.multiply(delta_s_t, C)
        # I = sigmoid(dot(W_i, x_t) + dot(U_i, h_t-1) + b_c)
        # caching this error: delta_i = delta_i * sigmoid_prime(I_args[bptt_step])
        delta_i = numpy.multiply(delta_i, af.sigmoid_prime(I_arg))
        dLdb_i += delta_i
        dLdU_i += numpy.outer(delta_i, H)
        dLdW_i += numpy.outer(delta_i, X)
        # need to update delta_h_t: each element += delta_i * U_i.T
        delta_h_t += numpy.dot(delta_i, self.U_i.T)

    def back_prop_candidate_gate(self, delta_s_t, delta_h_t, I, C_arg, H, X, dLdcs):
        dLdW_c, dLdU_c, dLdb_c = dLdcs

        # The "candidate gate." We know s_t = F * s_t-1 + I * C
        # so delta_c = delta_s_t * I. Both parts have shape (hidden_size)
        delta_c = numpy.multiply(delta_s_t, I)
        # C = f1(dot(W_c, x_t) + dot(U_c, h_t-1) + b_c)
        # caching this error: delta_c = delta_c * f1'(C_args[bptt_step])
        delta_c = numpy.multiply(delta_c, self.afunc_primes[0](C_arg))
        dLdb_c += delta_c
        # update dLdU_c: each element += delta_c * h_t-1..look at hf gate for dimensions
        dLdU_c += numpy.outer(delta_c, H)
        # update dLdW_c: each element += delta_c * x_t...look at hf gate for dimensions
        dLdW_c += numpy.outer(delta_c, X)
        # need to update delta_h_t: each element += U_c * delta_c
        # delta_c has shape (hidden_size), U_c has shape (hidden_size, hidden_size)
        delta_h_t += numpy.dot(delta_c, self.U_c.T)

    def back_prop_hidden_filter_gate(self, delta_s_t, delta_h_t, Hf_arg, H, X, dLdhfs):
        dLdW_hf, dLdU_hf, dLdb_hf = dLdhfs

        # four delta_h_t operations here:
        # The "hidden filter" gate: sigmoid(dot(W_hf, x_t) + dot(U_hf, h_t-1) + b_hf)
        # differentiating wrsp the argument gets us:
        # delta_h_t * sigmoid_prime(Hf_args[t]), each having size (hidden_size)
        delta_hf = numpy.multiply(delta_h_t, af.sigmoid_prime(Hf_arg))
        # time to update W_hf, U_hf, and b_hf
        dLdb_hf += delta_hf
        # each element of dLdU_hf should be += delta_hf * h_t-1
        # but delta_hf has shape (hidden_size), and h_t-1 has shape (hidden_size)
        # dLdU_hf has shape (hidden_size, hidden_size)
        dLdU_hf += numpy.outer(delta_hf, H)
        # each element of dLdW_hf should be += delta_hf * x_t
        # but delta_hf has shape (hidden_size), and x_t is (input_size)
        # dLdW_hf has shape (hidden_size, input_size)
        dLdW_hf += numpy.outer(delta_hf, X)
        # we also need to compute the first part of delta_h_t. This will be
        # the derivative of the "hf" gate wrsp to h_t-1:
        # delta_hf * U_hf.
        # delta_hf has shape (hidden_size), and U_hf (hidden_size, hidden_size).
        # So, we only need to compute delta_h_t = dot(delta_hf, U_hf.T)
        # the transpose is important. This is because when we multiplied forward:
        # [U_hf_00, U_hf_01, .., U_hf_0hidden] * [   h_t-10  ]
        # [U_hf_10, U_hf_11,      ...        ]   [   h_t-11  ]
        # [             ...                  ]   [    ...    ]
        # [U_hf_hidden0, U_hf_hidden1, ...   ]   [h_t-1hidden]

        # so, when we have error, we say delta_h_t has the errors for each element
        # in h_t due to h_t-1. So, we want to multiply the error for each component
        # by the one weight that contributed to predicting that element.
        # This can only be done by reorganizing U_f so that, when we perform the dot
        # operation, each row of delta_h_t will be multiplied by the row (now a column
        # in the transpose) that predicted that error unit in the forward direction.
        delta_h_t = numpy.dot(delta_h_t, self.U_hf.T)

    def init_gradients(self, delta, Hs):
        dLdW_f = numpy.zeros(self.W_f.shape)
        dLdU_f = numpy.zeros(self.U_f.shape)
        dLdb_f = numpy.zeros(self.b_f.shape)
        dLdfs = (dLdW_f, dLdU_f, dLdb_f)

        dLdW_i = numpy.zeros(self.W_i.shape)
        dLdU_i = numpy.zeros(self.U_i.shape)
        dLdb_i = numpy.zeros(self.b_i.shape)
        dLdis = (dLdW_i, dLdU_i, dLdb_i)

        dLdW_c = numpy.zeros(self.W_c.shape)
        dLdU_c = numpy.zeros(self.U_c.shape)
        dLdb_c = numpy.zeros(self.b_c.shape)
        dLdcs = (dLdW_c, dLdU_c, dLdb_c)

        dLdW_hf = numpy.zeros(self.W_hf.shape)
        dLdU_hf = numpy.zeros(self.U_hf.shape)
        dLdb_hf = numpy.zeros(self.b_hf.shape)
        dLdhfs = (dLdW_hf, dLdU_hf, dLdb_hf)

        dLdb_o = numpy.sum(delta, axis=0)

        # dLdW_o is more complex. The derivative for a single element is
        # h_t * delta, however, h_t has dimension (hidden_size) and
        # delta_h has dimensions (num_examples, output_size)
        # Hs on the other hand has dimensions (num_examples, hidden_size)
        # and dLdW_o should have dimensions (output_size, hidden_size)
        # so dLdW_o, to make dimensions agree, should be delta.T * Hs.
        # Remember, Hs has one extra row for the initial hidden state,
        # so we remove it (Hs[1:]) because that extra row is useless for gradient updates.
        dLdW_o = numpy.dot(delta.T, Hs[1:])
        return (dLdfs, dLdis, dLdcs, dLdhfs, (dLdW_o, dLdb_o))

    def feed_forward_and_cache(self, X):
        assert(X.shape[1] == self.input_size)
        time = X.shape[0]
        F_args = numpy.zeros((time, self.hidden_size))
        Fs = numpy.zeros(F_args.shape)

        I_args = numpy.zeros((time, self.hidden_size))
        Is = numpy.zeros(I_args.shape)

        C_args = numpy.zeros((time, self.hidden_size))
        Cs = numpy.zeros(C_args.shape)

        Hf_args = numpy.zeros((time, self.hidden_size))
        Hfs = numpy.zeros(Hf_args.shape)

        Hs = numpy.zeros((time+1, self.hidden_size))
        Hs[-1] = numpy.zeros(self.hidden_size)

        O_args = numpy.zeros((time, self.output_size))
        Os = numpy.zeros(O_args.shape)

        Ss = numpy.zeros((time+1, self.hidden_size))
        Ss[-1] = numpy.zeros(self.hidden_size)
        for t in range(time):
            # forward propagate, saving information as we go
            F_args[t] = numpy.dot(self.W_f, X[t]) + numpy.dot(self.U_f, Hs[t-1]) + self.b_f
            Fs[t] = af.sigmoid(F_args[t])

            I_args[t] = numpy.dot(self.W_i, X[t]) + numpy.dot(self.U_i, Hs[t-1]) + self.b_i
            Is[t] = af.sigmoid(I_args[t])

            C_args[t] = numpy.dot(self.W_c, X[t]) + numpy.dot(self.U_c, Hs[t-1]) + self.b_c
            Cs[t] = self.afuncs[0](C_args[t])

            Ss[t] = numpy.multiply(Fs[t], Ss[t-1]) + numpy.multiply(Is[t], Cs[t])

            Hf_args[t] = numpy.dot(self.W_hf, X[t]) + numpy.dot(self.U_hf, Hs[t-1]) + self.b_hf
            Hfs[t] = af.sigmoid(Hf_args[t])

            Hs[t] = numpy.multiply(Hfs[t], self.afuncs[1](Ss[t]))

            O_args[t] = numpy.dot(self.W_o, Hs[t]) + self.b_o
            Os[t] = self.afuncs[2](O_args[t])
        return F_args, Fs, I_args, Is, C_args, Cs, Hf_args, Hfs, Hs, O_args, Os, Ss

    def back_propagate_through_time(self, X, Y):
        time = X.shape[0]
        F_args, Fs, I_args, Is, C_args, Cs, Hf_args, Hfs, Hs, O_args, Os, Ss = self.feed_forward_and_cache(X)

        # (Os - Y) has dimensions (num_examples, output_size)
        # and O_args has dimensions (num_examples, output_size)
        # so delta has dimensions (num_examples, output_size)
        delta = numpy.multiply(Os - Y, self.afunc_primes[2](O_args))
        dLdfs, dLdis, dLdcs, dLdhfs, dLdOs = self.init_gradients(delta, Hs)
        for t in range(time):
            # to compute the error due to ht, element-wise, the equation is:
            # delta_h = delta * W_o.
            # A row of delta has dimensions (output_size) and W_o has
            # dimensions (output_size, hidden_size), so a row of delta_h must have dimensions
            # (hidden_size). This means delta_h = dot(delta[t], W_o)
            delta_ot_h = numpy.dot(delta[t], self.W_o)
            # delta_s_t will be the error that the "top data bus" that holds
            # s_(t-1) -> s_t transformation on it.
            # Likewise, delta_h_t will be the error that the "bottom data bus"
            # that holds the [h_(t-1), x_t] -> h_t transformation on it.
            # We need to compute this value before the back-through-time part
            # because initially, this error is only coming from o_t, while when
            # we start to go back through time, error along both busses will compound

            # We know h_t = h_f * f_2(s_t), so delta_s_t, initially should be
            # delta_h[t] * h_f * f'_2(s_t), each variable having dimensions (hidden_size)
            delta_s_t = numpy.multiply(numpy.multiply(delta_ot_h, Hfs[t]),
                                       self.afunc_primes[1](Ss[t]))
            # differentiating h_t w/ respect to h_f, we have:
            # delta_h[t] * f_2(s_t)
            delta_h_t = numpy.multiply(delta_ot_h, self.afuncs[1](Ss[t]))
 
            for bptt_step in range(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # We need to compute the four parts of delta_h_t here as well as the
                # one operation to compute delta_s_t which is really delta_s_t_minus_one

                self.back_prop_hidden_filter_gate(delta_s_t, delta_h_t, Hf_args[bptt_step],
                                                  Hs[bptt_step-1], X[bptt_step], dLdhfs)
                self.back_prop_candidate_gate(delta_s_t, delta_h_t, Is[bptt_step],
                                              C_args[bptt_step], Hs[bptt_step-1],
                                              X[bptt_step], dLdcs)
                self.back_prop_input_gate(delta_s_t, delta_h_t, Cs[bptt_step],
                                          I_args[bptt_step], Hs[bptt_step-1],
                                          X[bptt_step], dLdis)
                self.back_prop_forget_gate(delta_s_t, delta_h_t, Ss[bptt_step-1],
                                           F_args[bptt_step], Hs[bptt_step-1],
                                           X[bptt_step], dLdfs)

                # last update: need to update delta_s_t and delta_h_t so
                # that they get through the multiplication and f2 layer of the previous layer
                if bptt_step > 0:
                    # we know s_t = F * s_t-1 + I * C
                    # so delta_s_t-1 = delta_s_t * F
                    delta_s_t = numpy.multiply(delta_s_t, Fs[bptt_step])

                    # now we need to walk through f2 and mult layer to get final values for
                    # next iteration
                    # WE ARE NOW OPERATING ON THE END OF THE PREVIOUS LAYER
                    # we know h_t = Hf * f2(s_t)
                    # so, delta_s_t = delta_h_t * Hf * f2'(s_t)
                    delta_s_t += numpy.multiply(numpy.multiply(delta_h_t, Hfs[bptt_step-1]),
                                                self.afunc_primes[1](Ss[bptt_step-1]))
                    # and delta_h_t = delta_h_t * f2(s_t)
                    delta_h_t = numpy.multiply(delta_h_t, self.afuncs[1](Ss[bptt_step-1]))
        return dLdfs, dLdis, dLdcs, dLdhfs, dLdOs

    """
    def back_propagate_through_time_faster(self, X, Y):
        time = X.shape[0]
        F_args, Fs, I_args, Is, C_args, Cs, Hf_args, Hfs, Hs, O_args, Os, Ss = self.feed_forward_and_cache(X)
        delta = numpy.multiply(Os - Y, self.afunc_primes[2](O_args))
        dLdfs, dLdis, dLdcs, dLdhfs, dLdOs = self.init_gradients(delta, Hs)
        delta_s_t = numpy.zeros(self.hidden_size)
        delta_h_t = numpy.zeros(self.hidden_size)
        for t in range(time)[::-1]:
            delta_ot_h = numpy.dot(delta[t], self.W_o)
            delta_h_t += delta_ot_h
            delta_h_t = numpy.multiply(delta_h_t, self.afuncs[1](Ss[t]))
            delta_s_t += numpy.multiply(numpy.multiply(delta_h_t, Hfs[t]),
                                        self.afunc_primes[1](Ss[t]))
            self.back_prop_hidden_filter_gate(delta_s_t, delta_h_t, Hf_args[t],
                                              Hs[t-1], X[t], dLdhfs)
            self.back_prop_candidate_gate(delta_s_t, delta_h_t, Is[t], C_args[t], Hs[t-1],
                                          X[t], dLdcs)
            self.back_prop_input_gate(delta_s_t, delta_h_t, Cs[t], I_args[t], Hs[t-1],
                                      X[t], dLdis)
            self.back_prop_forget_gate(delta_s_t, delta_h_t, Ss[t-1], F_args[t], Hs[t-1],
                                       X[t], dLdfs)
            delta_s_t = numpy.multiply(delta_s_t, Fs[t])
        return dLdfs, dLdis, dLdcs, dLdhfs, dLdOs
    """

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

