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
import rnn


def main():
    num_features = 10
    output_size = 10
    hidden_size = 10

    X = numpy.zeros((10, 10))
    Y = numpy.zeros((10, 10))
    for i in range(10):
        X[i][i] = 1
        # if i+1 < output_size:
        Y[i][(i+1)%10] = 1

    # print(X)
    # print(Y)

    num_iterations = 400

    net = rnn.rnn(num_features, output_size, hidden_size=hidden_size, seed=10)
    for i in range(num_iterations):
        net.train([X], [Y])
        loss = net.loss_function([X], [Y])
        print(loss)
        if numpy.isnan(loss):
            #print(net.V)
            #print(net.W)
            #print(net.U)
            return
    # net.S = numpy.zeros(net.S.shape)
    #print(net.W)
    #print()
    #print(net.V)
    #print()
    #print(net.U)
    #print()

    # print(net.predict(X))
    #dist = net.predict_proba(X)
    #print(dist)
    #P = numpy.zeros(dist.shape)
    #P[range(X.shape[0]), numpy.argmax(dist, axis=1)] = 1
    #print(P)
    Y = net.predict(X)
    for x, y in zip(X, Y):
        print("%s -> %s" % (x, y))


if __name__ == "__main__":
    main()

