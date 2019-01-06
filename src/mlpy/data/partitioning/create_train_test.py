# SYSTEM IMPORTS
import numpy


# PYTHON PROJECT IMPORTS


def create_training_and_validation_data(X, Y, percent_in_validation_set):
    assert(X.shape[0] == Y.shape[0])

    unique_labels, counts = numpy.unique(Y, axis=0, return_counts=True)

    # split up data into "bins" for each label
    split_X = list()
    split_Y = list()

    for unique_label in unique_labels:
        indices = Y == unique_label
        split_X.append(X[indices])
        split_Y.append(Y[indices])

    train_X = list()
    train_Y = list()

    validation_X = list()
    validation_Y = list()

    for X_, Y_, in zip(split_X, split_Y):
        for x, y in zip(X, Y):
            if numpy.random.random_sample() <= percent_in_validation_set:
                validation_X.append(x)
                validation_Y.append(y)
            else:
                train_X.append(x)
                train_Y.append(y)

    train_X = numpy.array(train_X)
    train_Y = numpy.array(train_Y)

    validation_X = numpy.array(validation_X)
    validation_Y = numpy.array(validation_Y)

    assert(train_X.shape[0] + validation_X.shape[0] == X.shape[0])
    assert(train_Y.shape[0] + validation_Y.shape[0] == Y.shape[0])

    train_data = (train_X, train_Y,)
    validation_data = (validation_X, validation_Y,)

    return train_data, validation_data

