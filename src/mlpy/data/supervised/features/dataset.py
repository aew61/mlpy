# SYSTEM IMPORTS
import numpy
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_cd_, "..", "..")
for _dir_ in [_cd_, _src_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import feature_types as ftypes
import files
# import utils


__DATA_FILE_EXTENSION__ = ".data"


class Dataset(object):
    def __init__(self, data_file, fheader_obj, root_dir="."):
        self.fheader_obj = fheader_obj
        self.feature_value_map = self._build_feature_value_map(fheader_obj)
        self.annotation_value_func = self._build_annotation_func(fheader_obj)

        self.num_examples = 0
        self.num_features = len(fheader_obj.feature_names)

        # will have n rows for n examples (numpy ndarrays)
        self.X = None  # will have m columns for m features (n x m ndarray)
        self.Y = None  # will have 1 value for 1 annotation (n x 1 ndarray)

        self.data_file = data_file

        if not self.data_file.endswith(__DATA_FILE_EXTENSION__):
            self.data_file += __DATA_FILE_EXTENSION__

        self.root_dir = root_dir

    def feature_value(self, feature_name, feature_value):
        feature_value_dict = self.feature_value_map[feature_name]
        # print("name: %s, val: %s, dict: %s" % (feature_name, feature_value, feature_value_dict))
        if len(feature_value_dict) == 0:
            return float(feature_value)
        else:
            return feature_value_dict[feature_value]

    def _build_feature_value_map(self, fheader_obj):
        feature_map = dict()

        for f_name in fheader_obj.feature_names:
            # get feature type of feature
            f_type = fheader_obj.feature_type_map[f_name]
            f_vals = fheader_obj.feature_value_map[f_name]

            if f_type == ftypes.CONTINUOUS:
                feature_map[f_name] = dict()
            elif f_type == ftypes.NOMINAL or f_type == ftypes.ORDERED:
                feature_map[f_name] = {v: i for i, v in enumerate(f_vals)}
            else:
                raise ValueError("feature type [%s] is not recognized" % f_type)

        return feature_map

    def _build_annotation_func(self, fheader_obj):
        annotation_type = fheader_obj.annotation_type
        annotation_vals = fheader_obj.annotation_values

        if annotation_type == ftypes.CONTINUOUS:
            return lambda val: float(val)
        elif annotation_type == ftypes.NOMINAL or annotation_type == ftypes.ORDERED:
            return lambda val: {v: i for i, v in enumerate(annotation_vals)}[val]
        else:
            raise ValueError("annotation type [%s] is not recognized" % annotation_type)

    def _parse_data_file(self):
        data = list()
        # data_fpath = utils.find_file(self.data_file, root_dir=self.root_dir)
        # with open(data_fpath) as data_file:
        #     for line in data_file:
        #         line = utils.trim(line).strip()
        #         if len(line) > 0:
        #             data.append(line)
        data_fpath = files.find_file(self.data_file, root_dir=self.root_dir)
        with open(data_fpath) as data_file:
            for line in data_file:
                line = files.trim_line(line).strip()
                if len(line) > 0:
                    data.append(line)

        self.num_examples = len(data)

        self.X = numpy.zeros(tuple([self.num_examples, self.num_features]))
        self.Y = numpy.zeros(tuple([self.num_examples, 1]))

        for row_index, line in enumerate(data):
            feature_vals, annotation = self._parse_example(line)
            self.Y[row_index][0] = annotation
            for col_index, feature_val in enumerate(feature_vals):
                self.X[row_index][col_index] = feature_val

    def _parse_example(self, line):
        # return a list of feature values and the annotation
        elements = line.split(",")
        return [self._parse_value(e, f_name)
                for e, f_name in zip(elements[:-1], self.feature_value_map.keys())],\
               self._parse_value(elements[-1], "annotation")

    def _parse_value(self, val, name):
        if name == "annotation":
            return self.annotation_value_func(val)
        else:
            return self.feature_value(name, val)

    def delete_feature(self, feature_name):
        # we want to delete this feature from our dataset (for whatever reason),
        # we need to alter three 4 things:
        #   1) self.X (remove the column of that feature)
        #   2) self.num_features
        #   3) self.feature_value_map
        #   4) self.fheader_obj

        if feature_name in self.fheader_obj.feature_names:
            feature_col_index = self.fheader_obj.feature_names.index(feature_name)

            self.X = numpy.delete(self.X, feature_col_index, axis=1)
            self.num_features -= 1
            del self.feature_value_map[feature_name]
            self.fheader_obj.delete_feature(feature_name)

    def get_dataset(self):
        return self.fheader_obj.get_header(), self.X, self.Y, self

