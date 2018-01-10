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
import core
from data.formatting import create_folds


class StratifiedCrossValidator(core.Base):
    def __init__(self, num_folds, clf_type, feature_header={}):
        self.num_folds = num_folds
        self.feature_header = feature_header
        self.clf_type = clf_type
        self.clf_predictions = list()
        self.clf_expected_outputs = list()
        self.clfs = list()
        self.clf_args = {"feature_header": self.feature_header}

    def load(self, clf_type=None, clf_args={}):
        self.clf_args.update(clf_args)
        if clf_type is not None:
            self.clf_type = clf_type
        return self

    def _train(self, X, Y):
        folds = create_folds(X, Y, self.num_folds)
        for fold_num in range(self.num_folds):
            print("training fold: %s" % fold_num)
            test_X, test_Y = folds[fold_num]
            train_X = list()
            train_Y = list()

            print(" - building data")
            for i, (X_, Y_), in enumerate(folds):
                if i != fold_num:
                    train_X.append(X_)
                    train_Y.append(Y_)
            train_X = numpy.concatenate(tuple(train_X), axis=0)
            train_Y = numpy.concatenate(tuple(train_Y), axis=0)
            print("\tdone")

            print(" - training classifier")
            clf = self.clf_type(**self.clf_args).train(train_X, train_Y)
            print("\tdone")
            print(" - predicting data")
            self.clf_predictions.append(clf.predict(test_X))
            print("\tdone")
            self.clf_expected_outputs.append(test_Y)
            self.clfs.append(clf)
            

    def _predict_example(self, x):
        raise Exception("unsupported member function")

