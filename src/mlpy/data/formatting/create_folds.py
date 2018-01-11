# SYSTEM IMPORTS
import numpy


# PYTHON PROJECT IMPORTS


def create_folds(X, Y, num_folds):
    unique_ys, counts = numpy.unique(Y, return_counts=True)

    # each fold gets counts / num_folds examples (extra stuff gets shuffled randomly)
    num_y_per_fold = (counts / num_folds).astype(int)

    data_bins = list()
    for unique_y in unique_ys:
        indices = Y == unique_y
        new_X = X[indices]
        new_Y = Y[indices]

        new_Y = new_Y.reshape(new_Y.shape[0], 1)

        D = numpy.concatenate(tuple([new_X, new_Y]), axis=1)
        numpy.random.shuffle(D)

        data_bins.append(D)

    folds_X = [[] for _ in range(num_folds)]
    folds_Y = [[] for _ in range(num_folds)]

    max_index_assigned_per_y = numpy.zeros(num_y_per_fold.shape[0], dtype=int)

    for y_index, D in enumerate(data_bins):
        X_ = D[:, :-1]
        Y_ = D[:, -1]

        num_elements = num_y_per_fold[y_index]
        for i in range(num_folds):
            folds_X[i].append(X_[i * num_elements: (i+1) * num_elements])
            folds_Y[i].append(Y_[i * num_elements: (i+1) * num_elements])

        max_index_assigned_per_y[y_index] = num_folds * num_elements

    # now go through each bin, and distribute the data that is left into each fold
    for y_index, D in enumerate(data_bins):
        X_ = D[:, :-1][max_index_assigned_per_y[y_index]:]
        Y_ = D[:, -1][max_index_assigned_per_y[y_index]:]

        for x, y in zip(X_, Y_):
            fold_index = numpy.random.randint(0, high=num_folds)
            folds_X[fold_index].append(x.reshape(1, x.shape[0]))  # this is a vector
            folds_Y[fold_index].append(numpy.array([y]))  # this is a scalar...make it a vector

    folds = list()
    for fold_X, fold_Y in zip(folds_X, folds_Y):
        X_concat = numpy.concatenate(tuple(fold_X), axis=0)
        Y_concat = numpy.concatenate(tuple(fold_Y))

        folds.append((X_concat, Y_concat,))

    # one last thing....shuffle each fold so we don't get "groupings" of data that are consistent
    # each time this function is called

    total_rows = 0

    for i, (X_, Y_) in enumerate(folds):
        Y_ = Y_.reshape(Y_.shape[0], 1)
        D = numpy.concatenate(tuple([X_, Y_]), axis=1)
        numpy.random.shuffle(D)
        folds[i] = (D[:, :-1], D[:, -1],)
        assert(folds[i][0].shape[0] == folds[i][1].shape[0])
        assert(len(folds[i][0].shape) == len(X.shape) and len(folds[i][1].shape) == len(Y.shape))
        for s1, s2 in zip(folds[i][0].shape[1:], X.shape[1:]):
            assert(s1 == s2)
        for s1, s2 in zip(folds[i][1].shape[1:], Y.shape[1:]):
            assert(s1 == s2)
        total_rows += folds[i][0].shape[0]

    return folds

