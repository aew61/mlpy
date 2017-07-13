# SYSTEM IMPORTS
import matplotlib.pyplot as plt
import numpy
import os
import random
import sys


_current_dir_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_current_dir_, "..", "..", "..")
_dirs_to_add_ = [_current_dir_, _src_dir_]
for _dir_ in _dirs_to_add_:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _dirs_to_add_
del _src_dir_
del _current_dir_


# PYTHON PROJECT IMPORTS
from mlpy.nets import elm

def sinc(x):
    # the sinc function: y(x) = {sin(x) / x if x != 0, 1 otherwise}
    y = numpy.sin(x) / x
    y[numpy.isnan(y)] = 1.0
    return y


def create_training_set(num_examples):
    x_values = numpy.array([random.uniform(-10.0, 10.0) for x in range(num_examples)],
                           dtype=float).reshape((num_examples, 1))
    y_values = sinc(x_values)

    # add random noise to training examples
    y_values += numpy.array([random.uniform(-0.5, 0.5) for x in range(num_examples)],
                            dtype=float).reshape((num_examples, 1))
    return x_values, y_values


def create_validation_set(num_examples):
    x_values = numpy.array([random.uniform(-10.0, 10.0) for x in range(num_examples)],
                           dtype=float).reshape((num_examples, 1))

    return x_values, sinc(x_values)


def main():
   num_examples = 50000
   training_examples, training_annotations = create_training_set(num_examples)
   validation_examples, validation_annotations = create_validation_set(num_examples)

   num_hidden_neurons = 20
   layers = [1, num_hidden_neurons, 1]
   classifier = elm(layers)

   classifier.train(training_examples, training_annotations)
   outputs = classifier.classify(validation_examples)

   plt.xlabel('x')
   plt.ylabel('sin(c) := sin(x) / x if x != 0, 1 otherwise')
   training = plt.plot(training_examples, training_annotations, 'bo', label='noisy train. data')
   expected = plt.plot(validation_examples, validation_annotations, 'r+', label='sinc(x)',
                       markersize=2)
   actual = plt.plot(validation_examples, outputs, 'g+', label='elm(x)', markersize=2)
   plt.legend()
   plt.show()


if __name__ == "__main__":
    main()

