# SYSTEM IMPORTS
import time
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
import id3
import leid3
import ppyid3


def main():
    name_list = ["voting data", "volcanoe data", "spam data"]
    data_list = [data.load_voting_data(), data.load_volcanoe_data(), data.load_spam_data()]

    clf_name_list = ["opt id3", "  leid3", " ppyid3"]
    clf_type_list = [id3.ID3Tree, leid3.LEID3Tree, ppyid3.PPYID3Tree]
    for name, dataset in zip(name_list, data_list):
        print(name)
        h, X, Y, _ = dataset
        for clf_name, clf_type in zip(clf_name_list, clf_type_list):
            t = clf_type(feature_header=h)
            t1 = time.time()
            t.train(X, Y)
            t2 = time.time() - t1
            if t2 > 60:
                print(" - {0} took {1:.3}min to train with {2:.4} accuracy".format(clf_name, t2 / 60, numpy.sum(t.predict(X) == Y) / Y.shape[0]))
            else:
                print(" - {0} took {1:.3}sec to train with {2:.4} accuracy".format(clf_name, t2, numpy.sum(t.predict(X) == Y) / Y.shape[0]))
            # if clf_name == "opt id3":
            #     cont_lt_count = 0.0
            #     cont_lt_total = 0.0
            #     for n in t.tree_impl.interiors():
            #         if hasattr(n.data, "lt_partition_value"):
            #             cont_lt_total += 1.0
            #             if n.data.lt_partition_value is not None:
            #                 cont_lt_count += 1.0
            #     if cont_lt_total == 0.0:
            #         cont_lt_total = 1.0
            #     print(" -- opt id3 had {0:.3f} of continuous features with >= case".format(cont_lt_count / cont_lt_total))


if __name__ == "__main__":
    main()

