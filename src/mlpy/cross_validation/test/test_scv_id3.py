# SYSTEM IMPORTS
import numpy
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_cd_, "..")
_module_dir_ = os.path.join(_src_dir_, "..")
for _dir_ in [_cd_, _src_dir_, _module_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _module_dir_
del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import data
import trees
import scv


def main():
    h, X, Y, _ = data.load_voting_data()
    d = trees.id3.ID3DTree
    num_folds = 5
    max_depth = 3
    gain_ratio = True

    clf_args = {"feature_header": h, "max_depth": max_depth, "use_gain_ratio": gain_ratio}

    c = scv.StratifiedCrossValidator(num_folds, d, feature_header=h).load(clf_args=clf_args).train(X, Y)
    for Y_pred, Y_act in zip(c.clf_predictions, c.clf_expected_outputs):
        print(numpy.sum(numpy.array(Y_pred) == Y_act) / Y_act.shape[0])


if __name__ == "__main__":
    main()

