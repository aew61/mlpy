# SYSTEM IMPORTS
import matplotlib.pyplot as plt
import numpy
import os
import random
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_scripts_dir_ = os.path.join(_cd_, "..", "..")
for _dir_ in [_cd_, _scripts_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _scripts_dir_
del _cd_


# PYTHON PROJECT IMPORTS
from nets import ANN


def main():
    num_examples = 3
    num_features = 2
    num_outputs = 1
    learning_rate = 0.005
    weight_decay_coeff = 0.0
    X = numpy.array([[3, 5],
                     [5, 1],
                     [10, 2]], dtype=float)
    Y = numpy.array([[75], [82], [93]], dtype=float)

    X /= numpy.amax(X, axis=0)
    Y /= 100

    print("X:\n%s" % X)
    print("Y:\n%s" % Y)
    print()

    # make the neural net
    net = ANN([num_features, 3, num_outputs], learning_rate=learning_rate,
              weight_decay_coeff=weight_decay_coeff)

    print(net.predict(X))

    print("---------------")
    print("weights shapes:")
    for w in net.weights:
        print(w.shape)
    print("---------------")
    print()

    validation_features = numpy.array([[0.8, 0.6]])
    # validation_features = numpy.array([[8, 3]])

    costs = list()
    num_iterations = 10000
    iters = range(num_iterations)

    for i in iters:
        net.train(X, Y)
        costs.append(net.cost_function(X, Y))

    print("validation set:\n%s" % validation_features)
    print("validation:\n%s" % net.predict(validation_features))

    # plt.plot(iters, costs)
    # plt.show()

if __name__ == "__main__":
    main()

