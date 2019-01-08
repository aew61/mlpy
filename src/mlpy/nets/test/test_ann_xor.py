# SYSTEM IMPORTS
# import matplotlib.pyplot as plt
import numpy
import os
import random
import sys


_current_dir_ = os.path.abspath(os.path.dirname(__file__))
_scripts_dir_ = os.path.join(_current_dir_, "..", "..")
_dirs_to_add_ = list([_current_dir_, _scripts_dir_])
for _dir_ in _dirs_to_add_:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _dirs_to_add_
del _scripts_dir_
del _current_dir_


# PYTHON PROJECT IMPORTS
from nets import ann

def main():
    num_examples = 4
    num_features = 2
    num_outputs = 1
    learning_rate = 0.05
    weight_decay_coeff = 0.0
    training_features = numpy.array([[0,0],
                                     [0,1],
                                     [1,0],
                                     [1,1]], dtype=float)
    training_labels = numpy.array([[0],
                                   [1],
                                   [1],
                                   [0]], dtype=float)

    # training_features /= numpy.amax(training_features, axis=0)
    # training_labels /= 100

    print("training_features:\n%s" % training_features)
    print("training_labels:\n%s" % training_labels)

    # make the neural net
    net = ann([num_features, 5, num_outputs], learning_rate=learning_rate, weight_decay_coeff=weight_decay_coeff)

    # validation_features = numpy.array([[0, 0, 0]])

    costs = list()
    num_iterations = 10000
    iters = list([i for i in range(num_iterations)])

    for i in iters:
        net.train(training_features, training_labels)
        # if i % 1000 == 0:
        #     print("training iteration %s" % i)
        #     print("validation set:%s" % validation_features)
        #     print("validation:%s" % net.feed_forward(validation_features))
        print(net.loss_function(training_features, training_labels))

    print(net.predict(training_features))

    # print("validation set:%s" % validation_features)
    # print("validation:%s" % net.predict(validation_features))

    # lines = plt.plot(iters, costs)
    # plt.show()

if __name__ == "__main__":
    main()

