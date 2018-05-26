# SYSTEM IMPORTS
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
from nets import elm


def main():
    num_examples = 3
    num_features = 2
    num_outputs = 1
    learning_rate = 0.005
    weight_decay_coeff = 0.0
    training_features = numpy.array([[3, 5],
                                     [5, 1],
                                     [10, 2]], dtype=float)
    training_labels = numpy.array([[75],
                                   [82],
                                   [93]], dtype=float)

    # training_features /= numpy.amax(training_features, axis=0)
    training_labels /= 100

    print("training_features:\n%s" % training_features)
    print("training_labels:\n%s" % training_labels)

    # make the neural net
    net = elm([num_features, 3, num_outputs])

    validation_features = numpy.array([[8, 3]])

    costs = list()
    num_iterations = 10000
    iters = range(num_iterations)

    # for i in iters:
        # print("training iteration %s" % i)
    net.train(training_features, training_labels)
        # print("validation set:\n%s" % validation_features)
        # print("validation:\n%s" % net.feed_forward(validation_features))
        # costs.append(net.cost_function(training_features, training_labels,
        #                                weight_decay_coeff=weight_decay_coeff))

    print("validation set:\n%s" % validation_features)
    print("validation:\n%s" % net.predict(validation_features))

    # plt.plot(iters, costs)
    # plt.show()

if __name__ == "__main__":
    main()

