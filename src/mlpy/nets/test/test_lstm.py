# SYSTEM IMPORTS
import numpy
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_net_dir_ = os.path.join(_cd_, "..")
for _dir_ in [_cd_, _net_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _net_dir_
del _cd_


# PYTHON PROJECT IMPORTS
from lstm import lstm


def main():
    num_features = 10
    output_size = 10

    X = numpy.zeros((10, 10))
    Y = numpy.zeros((10, 10))
    for i in range(10):
        X[i][i] = 1
        # if i+1 < output_size:
        Y[i][(i+1)%10] = 1

    # print(X)
    # print(Y)

    num_iterations = 3000

    net = lstm(num_features, output_size, seed=10)
    print(net.predict(X))

    """"""
    for i in range(num_iterations):
        net.train([X], [Y])
        # print(net.predict(X))
        loss = net.loss_function([X], [Y])
        print(loss)
        
        if numpy.isnan(loss):
            # """"""
            print(net.W_f)
            print(net.W_i)
            print(net.W_c)
            print(net.W_hf)
            print(net.W_o)
            # """"""
            # """"""
            print(net.b_f)
            print(net.b_i)
            print(net.b_c)
            print(net.b_hf)
            print(net.b_o)
            # """"""
            return
    """"""
    # net.S = numpy.zeros(net.S.shape)
    #print(net.W)
    #print()
    #print(net.V)
    #print()
    #print(net.U)
    #print()

    # print(net.F_w)
    # print(net.I_w)
    # print(net.C_w)
    # print(net.Of_w)

    # print(net.F_b)
    # print(net.I_b)
    # print(net.C_b)
    # print(net.Of_b)
    net.reset()
    # print(net.predict(X))
    Y = net.predict(X)
    for x, y in zip(X, Y):
        print("%s -> %s" % (x, y))


if __name__ == "__main__":
    main()

