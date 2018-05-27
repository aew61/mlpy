# SYSTEM IMPORTS
import numpy
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
# _src_dir_ = os.path.join(_cd_, "..")
for _dir_ in [_cd_]: # , _src_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
# del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import feature_types as ftypes


def make_one_hot_discrete_dataset(dataset, and_annotations=False):
    # need to remake the header, the data, and optionally the annotations

    new_F = None
    new_X = list()
    index = 0
    for col, f_name in enumerate(dataset.fheader_obj.feature_names):

        f_type = dataset.fheader_obj.feature_type_map[f_name]
        if ftypes.is_discrete(f_type):
            new_F = make_one_hot_discrete_data(dataset.X[:, col], f_type)
        else:
            new_F = dataset.X[:, col]
            new_F = new_F.reshape(new_F.shape[0], 1)

        new_X.append(new_F)
        dataset.fheader_obj.feature_index_map[f_name] = index

        if len(new_F.shape) == 1:
            index += 1
        else:
            index += new_F.shape[1]

    dataset.X = numpy.concatenate(tuple(new_X), axis=1)

    if and_annotations and ftypes.is_discrete(dataset.fheader_obj.annotation_type):
        dataset.Y = make_one_hot_discrete_data(dataset.Y, dataset.fheader_obj.annotation_type)

    return dataset.get_dataset()

def make_one_hot_discrete_data(F, f_type):
    if not ftypes.is_discrete(f_type):
        raise TypeError("feature type [%s] is not discrete" % f_type)

    sorted_unique_values = numpy.sort(numpy.unique(F))

    one_hot_dict = {v: i for i, v in enumerate(sorted_unique_values)}

    new_F = numpy.zeros((F.shape[0], sorted_unique_values.shape[0]))

    for i in range(F.shape[0]):
        new_F[i][one_hot_dict[F[i]]] = 1

    return new_F

