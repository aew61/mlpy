# SYSTEM IMPORTS
import functools
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


def run_cv(folds, clf_funcs, headers, datasets):
    assert(len(folds) == len(clf_funcs) and len(clf_funcs) == len(headers) and len(headers) == len(datasets))

    scvs = list()
    for i, (num_folds, clf_func, h, (X, Y)), in enumerate(zip(folds, clf_funcs, headers, datasets)):
        print("cv experiment %s/%s" % (i, len(folds)))
        scvs.append(scv.StratifiedCrossValidator(num_folds, clf_func).train(X, Y))
    return scvs


def base_id3_instantiation(args, fold_num, max_folds):
        return trees.id3.ID3DTree(**args)


def part_a(voting_data, volcanoe_data, spam_data):
    print("part_a")
    num_folds = 5
    max_depth = 1
    use_gain_ratio = False

    h_vote, X_vote, Y_vote, _ = voting_data
    h_volcanoe, X_volcanoe, Y_volcanoe, _ = volcanoe_data
    h_spam, X_spam, Y_spam, _ = spam_data

    data_names = ["voting data", "volcanoe data", "spam data"]
    header_list = [h_vote, h_volcanoe, h_spam]
    dataset_list = [(X_vote, Y_vote,), (X_volcanoe, Y_volcanoe,), (X_spam, Y_spam,)]
    folds_list = [num_folds, num_folds, num_folds]

    funcs = list()
    for h in header_list:
        args = {"feature_header": h, "max_depth": max_depth, "use_gain_ratio": use_gain_ratio}
        funcs.append(functools.partial(base_id3_instantiation, args))

    # now run the cv experiment
    scvs = run_cv(folds_list, funcs, header_list, dataset_list)

    for s, name in zip(scvs, data_names):
        # compute average accuracy
        total_accuracy = 0
        for Y_pred, Y_act in zip(s.clf_predictions, s.clf_expected_outputs):
            total_accuracy += numpy.sum(numpy.array(Y_pred) == Y_act) / Y_act.shape[0]
        total_accuracy /= len(s.clf_predictions)
        print("accuracy for id3 with max_depth [%s] on [%s] is [%s]" % (max_depth, name, total_accuracy))
    print()


def part_b(volcanoe_data, spam_data):
    print("part_b")
    h_volcanoe, X_volcanoe, Y_volcanoe, dataset_obj_volcanoe = volcanoe_data
    h_spam, X_spam, Y_spam, dataset_obj_spam = spam_data

    names_list = ["volcanoe data", "spam data"]
    header_list = [h_volcanoe, h_spam]
    data_list = [(X_volcanoe, Y_volcanoe,), (X_spam, Y_spam,)]
    dataset_obj_list = [dataset_obj_volcanoe, dataset_obj_spam]

    for name, h, (X, Y), dataset_obj in zip(names_list, header_list, data_list, dataset_obj_list):
        d = trees.id3.ID3DTree()
        max_ig_feature_index, max_ig = d.max_information_gain(X, Y)

        # get type of feature id and the name of the feature id
        feature_type = h[max_ig_feature_index]
        feature_name = {i: name for name, i in dataset_obj.fheader_obj.feature_index_map.items()}[max_ig_feature_index]

        print("1st feature picked for [%s] is [%s] with type [%s]" % (name, feature_name, feature_type))
    print()


def part_c(volcanoe_data, spam_data):
    print("part_c")
    num_folds = 5
    max_depth_list = [i + 1 for i in range(10)] + [numpy.inf]
    use_gain_ratio = False

    h_volcanoe, X_volcanoe, Y_volcanoe, _ = volcanoe_data
    h_spam, X_spam, Y_spam, _ = spam_data

    data_names = ["volcanoe data", "spam data"]
    header_list = [h_volcanoe, h_spam]
    dataset_list = [(X_volcanoe, Y_volcanoe,), (X_spam, Y_spam,)]
    folds_list = [num_folds, num_folds, num_folds]

    funcs = list()
    for h in header_list:
        args = {"feature_header": h, "max_depth": max_depth, "use_gain_ratio": use_gain_ratio}
        funcs.append(functools.partial(base_id3_instantiation, args))

    # now run the cv experiment
    scvs = run_cv(folds_list, funcs, header_list, dataset_list)
    print()



def main():
    voting_data = data.load_voting_data()
    volcanoe_data = data.load_volcanoe_data()
    spam_data = data.load_spam_data()

    part_a(voting_data, volcanoe_data, spam_data)
    part_b(volcanoe_data, spam_data)


if __name__ == "__main__":
    main()

