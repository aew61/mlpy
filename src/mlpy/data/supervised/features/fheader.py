# SYSTEM IMPORTS
import os
import re
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_cd_, "..", "..")
for _dir_ in [_cd_, _src_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
import feature_types as ftypes
import utils


__NAMES_FILE_EXTENSION__ = ".names"


# parse a C4.5 .names file
class FHeader(object):
    def __init__(self, names_file, root_dir="."):
        self.feature_names = list()
        self.feature_index_map = dict()
        self.feature_value_map = dict()
        self.feature_type_map = dict()
        self.annotation_type = None
        self.annotation_values = None
        self.names_file = names_file

        if not self.names_file.endswith(__NAMES_FILE_EXTENSION__):
            self.names_file += __NAMES_FILE_EXTENSION__

        self.root_dir = root_dir

    def _parse_feature(self, line):
        line = utils.trim(line)

        if len(line) == 0:
            return None, None, None

        # check if there is no colon character (class line)
        colon_index = line.find(":")
        if colon_index < 0:
            return ("annotation",) + self._parse_feature_values(line.strip())
        return (line[:colon_index].strip(),) + self._parse_feature_values(line[colon_index+1:].strip())

    def _parse_feature_values(self, line):
        # line could be "continuous"
        if line.lower() == "continuous":
            return ftypes.CONTINUOUS, list()

        f_type = ftypes.NOMINAL
        f_vals = list()
        for v in line.split(","):
            v = v.strip()
            # make sure there are no quotes surrounding the value
            if len(v) >= 0 and v[0] == "\"" and v[-1] == "\"":
                v = v[1:-1].strip()
            f_vals.append(v)
        return f_type, sorted(f_vals)

    def create_header(self):
        f_path = utils.find_file(self.names_file, self.root_dir)

        index = 0
        # now parse the .names file....first line should be the class
        with open(f_path, "r") as header_file:
            for i, line in enumerate(header_file):
                feature_name, feature_type, feature_values = self._parse_feature(line)
                if feature_name is not None:
                    if feature_name == "annotation":
                        self.annotation_type = feature_type
                        self.annotation_values = feature_values
                    else:
                        # add to dictionaries

                        if feature_name in self.feature_names:
                            raise ValueError("DUPLICATE FEATURE: feature [%s] already exists!" % feature_name)

                        self.feature_names.append(feature_name)
                        self.feature_value_map[feature_name] = feature_values
                        self.feature_type_map[feature_name] = feature_type
                        self.feature_index_map[feature_name] = index
                        index += 1

    def get_feature_names(self):
        return list(self.feature_names)

    def get_header(self):
        return {i: self.feature_type_map[f_name] for f_name, i in self.feature_index_map.items()}
        # return {i: self.feature_type_map[f_name] for i, f_name in enumerate(self.feature_names)}

    def delete_feature(self, feature_name):
        if feature_name in self.feature_names:
            # need to update feature_names
            # need to update feature_value_map
            # need to update feature_type_map

            self.feature_names.remove(feature_name)
            del self.feature_value_map[feature_name]
            del self.feature_type_map[feature_name]

            index = self.feature_index_map[feature_name]
            del self.feature_index_map[feature_name]
            # need to adjust all other indexes to reflect the change
            for f_name in self.feature_index_map.keys():
                if self.feature_index_map[f_name] > index:
                    self.feature_index_map[f_name] -= 1

