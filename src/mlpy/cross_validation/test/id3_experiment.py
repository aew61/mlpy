# SYSTEM IMPORTS
import collections
import functools
import matplotlib.pyplot as plt
import multiprocessing
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


cd = os.path.abspath(os.path.dirname(__file__))


def train_and_return_avg_accuracy(num_folds, clf_func, X, Y, cv, child_pipe):
    predictions = None
    expected = None

    if cv:
        clf = scv.StratifiedCrossValidator(num_folds, clf_func).train(X, Y)
        predicted = clf.clf_predictions
        expected = clf.clf_expected_outputs
    else:
        clf = clf_func(0, num_folds).train(X, Y)
        predicted = [clf.predict(X)]
        expected = [Y]

    # compute average accuracy
    total_accuracy = 0
    for Y_pred, Y_act in zip(predicted, expected):
        total_accuracy += numpy.sum(numpy.array(Y_pred) == Y_act) / Y_act.shape[0]
    total_accuracy /= len(expected)
    child_pipe.send(total_accuracy)
    child_pipe.close()
    print(" - done with cv experiment")
    return


def run_experiment(folds, clf_funcs, headers, datasets, cv=True):
    assert(len(folds) == len(clf_funcs) and len(clf_funcs) == len(headers) and len(headers) == len(datasets))

    processes = list()
    parent_pipes = list()
    child_pipes = list()
    for i, (num_folds, clf_func, h, (X, Y)), in enumerate(zip(folds, clf_funcs, headers, datasets)):
        print("cv experiment %s/%s" % (i, len(folds)))
        parent_pipe, child_pipe = multiprocessing.Pipe()
        p = multiprocessing.Process(target=train_and_return_avg_accuracy,
            args=(num_folds, clf_func, X, Y, cv, child_pipe,))
        processes.append(p)
        parent_pipes.append(parent_pipe)
        child_pipes.append(child_pipe)

    output_accuracies = list()

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print(" - all processes joined")
    for parent_pipe in parent_pipes:
        output_accuracies.append(parent_pipe.recv())
    print(" - all accuracies received")

    return output_accuracies


def base_id3_instantiation(h, d, ugr, fold_num, max_folds):
        return trees.id3.ID3DTree(feature_header=h, max_depth=d, use_gain_ratio=ugr)


def plot_accuracies_vs_max_depth(max_depths, name_data_pairs, name="", save=True):
    lines = list()
    for line_name, line_ys in name_data_pairs:
        lines.append(plt.plot(max_depths, line_ys, label=line_name))
    plt.ylabel("avg 5 fold CV ID3 accuracy (%)")
    plt.xlabel("max depth of ID3 tree")
    plt.legend()

    if not save:
        plt.show()
    else:
        plt.savefig(os.path.join(cd, name))


def plot_percent_diff_vs_max_depth(max_depths, name_data_pairs, name="", save=True):
    lines = list()
    for line_name, line_ys in name_data_pairs:
        lines.append(plt.plot(max_depths, line_ys, label=line_name))
    plt.ylabel("percent diff between 5 fold CV and full sample accuracies")
    plt.xlabel("max depth of ID3 tree")
    plt.legend()

    if not save:
        plt.show()
    else:
        plt.savefig(os.path.join(cd, name))


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
        funcs.append(functools.partial(base_id3_instantiation, h, max_depth, use_gain_ratio))

    # now run the cv experiment
    output_accuracies = run_experiment(folds_list, funcs, header_list, dataset_list)

    for output_accuracy, name in zip(output_accuracies, data_names):
        print("accuracy for id3 with max_depth [%s] on [%s] is [%s]" % (max_depth, name, output_accuracy))
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
    max_depth_list = [i + 1 for i in range(10)]
    use_gain_ratio = False

    h_volcanoe, X_volcanoe, Y_volcanoe, _ = volcanoe_data
    h_spam, X_spam, Y_spam, _ = spam_data

    data_names = ["volcanoe data", "spam data"]
    header_list = [h_volcanoe, h_spam]
    dataset_list = [(X_volcanoe, Y_volcanoe,), (X_spam, Y_spam,)]
    folds_list = [num_folds, num_folds]

    func_matrix = list()

    volcanoe_accuracies = list()
    spam_accuracies = list()

    for i in range(len(max_depth_list)):

        funcs = list()
        for h in header_list:
            funcs.append(functools.partial(base_id3_instantiation, dict(h),
                int(max_depth_list[i]), bool(use_gain_ratio)))

        func_matrix.append(funcs)

    for func_list in func_matrix:
        print("running cv experiments with max depth: %s" % func_list[0].args[1])
        # now run the cv experiment
        output_accuracies = run_experiment(folds_list, func_list, header_list, dataset_list)
        volcanoe_accuracies.append(output_accuracies[0])
        spam_accuracies.append(output_accuracies[1])
        print(" - done")

    plot_accuracies_vs_max_depth(max_depth_list, [(n, a,) for n, a in zip(data_names, [volcanoe_accuracies, spam_accuracies])], save=True, name="part_c.png")
    print()


def part_d(voting_data, volcanoe_data, spam_data):
    print("part_d")
    num_folds = 5
    max_depth_list = [1, 3, 5]
    use_gain_ratio_list = [False, True]

    h_voting, X_voting, Y_voting, _ = voting_data
    h_volcanoe, X_volcanoe, Y_volcanoe, _ = volcanoe_data
    h_spam, X_spam, Y_spam, _ = spam_data

    dataset_names = ["voting data", "volcanoe data", "spam data"]
    header_list = [h_voting, h_volcanoe, h_spam]
    dataset_list = [(X_voting, Y_voting,), (X_volcanoe, Y_volcanoe,), (X_spam, Y_spam,)]
    folds_list = [num_folds, num_folds, num_folds]

    func_matrix = list()
    voting_accuracies = list()
    volcanoe_accuracies = list()
    spam_accuracies = list()

    for depth in max_depth_list:
        for ugr in use_gain_ratio_list:
            funcs = list()
            for h in header_list:
                funcs.append(functools.partial(base_id3_instantiation, dict(h),
                    int(depth), bool(ugr)))
            func_matrix.append(funcs)

    accuracies_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    i = 0
    for depth in max_depth_list:
        for ugr in use_gain_ratio_list:
            output_accuracies = run_experiment(folds_list, func_matrix[i], header_list, dataset_list)
            for n, accuracy in zip(dataset_names, output_accuracies):
                accuracies_dict[ugr][n + " gain ratio=%s" % ugr].append(accuracy)
            i += 1

    name_accuracy_tuples = list()
    for val in accuracies_dict.values():
        name_accuracy_tuples += [(n, a,) for n, a in val.items()]

    plot_accuracies_vs_max_depth(max_depth_list, name_accuracy_tuples, save=True, name="part_d.png")
    print()


def part_e(voting_data, volcanoe_data, spam_data):
    print("part_e")
    num_folds = 5
    max_depth_list = [1, 2]
    use_gain_ratio = False

    h_voting, X_voting, Y_voting, _ = voting_data
    h_volcanoe, X_volcanoe, Y_volcanoe, _ = volcanoe_data
    h_spam, X_spam, Y_spam, _ = spam_data

    dataset_names = ["voting data", "volcanoe data", "spam data"]
    header_list = [h_voting, h_volcanoe, h_spam]
    dataset_list = [(X_voting, Y_voting,), (X_volcanoe, Y_volcanoe,), (X_spam, Y_spam,)]
    folds_list = [num_folds, num_folds, num_folds]

    func_matrix = list()
    voting_cv_accuracies = list()
    volcanoe_cv_accuracies = list()
    spam_cv_accuracies = list()

    for i in range(len(max_depth_list)):
        funcs = list()
        for h in header_list:
            funcs.append(functools.partial(base_id3_instantiation, dict(h),
                int(max_depth_list[i]), bool(use_gain_ratio)))

        func_matrix.append(funcs)

    for func_list in func_matrix:
        print("running cv experiments with max depth: %s" % func_list[0].args[1])
        # now run the cv experiment
        output_accuracies = run_experiment(folds_list, func_list, header_list, dataset_list)
        voting_cv_accuracies.append(output_accuracies[0])
        volcanoe_cv_accuracies.append(output_accuracies[1])
        spam_cv_accuracies.append(output_accuracies[2])
        print(" - done")

    # now do full sample training and prediction for both depths
    voting_full_accuracies = list()
    volcanoe_full_accuracies = list()
    spam_full_accuracies = list()

    for func_list in func_matrix:
        print("running cv experiments with max depth: %s" % func_list[0].args[1])
        output_accuracies = run_experiment(folds_list, func_list, header_list, dataset_list, cv=False)
        voting_full_accuracies.append(output_accuracies[0])
        volcanoe_full_accuracies.append(output_accuracies[1])
        spam_full_accuracies.append(output_accuracies[2])

    aligned_accuracies = [(voting_cv_accuracies, voting_full_accuracies,),
                          (volcanoe_cv_accuracies, volcanoe_full_accuracies,),
                          (spam_cv_accuracies, spam_full_accuracies,)]

    aligned_accuracy_percent_diffs = list()
    for cv_acc, full_acc in aligned_accuracies:
        aligned_accuracy_percent_diffs.append(numpy.abs(numpy.array(cv_acc) - numpy.array(full_acc)) / 100)

    name_val_tuples = [(n, a) for n, a in zip(dataset_names, aligned_accuracy_percent_diffs)]

    plot_percent_diff_vs_max_depth(max_depth_list, name_val_tuples, save=True, name="part_e.png")
    print()


def main():
    voting_data = data.load_voting_data()
    volcanoe_data = data.load_volcanoe_data()
    spam_data = data.load_spam_data()

    part_a(voting_data, volcanoe_data, spam_data)
    part_b(volcanoe_data, spam_data)
    part_c(volcanoe_data, spam_data)
    part_d(voting_data, volcanoe_data, spam_data)
    part_e(voting_data, volcanoe_data, spam_data)


if __name__ == "__main__":
    main()

