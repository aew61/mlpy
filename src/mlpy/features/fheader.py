# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
import ftypes


__NAMES_FILE_EXTENSION__ = ".names"


# parse a C4.5 .names file
class FHeader(object):
    def __init__(self, names_file, root_dir="."):
        self.features_map = dict()
        self.feature_type_map = dict()
        self.names_file = names_file

        if not self.names_file.endswith(__NAMES_FILE_EXTENSION__):
            self.names_file += __NAMES_FILE_EXTENSION__

        self.root_dir = root_dir

    def _find_file(self, f_name):
        # look for f_name in root_dir and subdirectories
        file_path = None
        if os.path.exists(f_name):
            file_path = f_name
        else:
            for dir_path, _, _ in os.walk(self.root_dir, topdown=True):
                if os.path.exists(os.path.join(dir_path, f_name)):
                    return os.path.join(dir_path, f_name)

        if file_path is None:
            raise ValueError("Could not find file [%s] in root_dir [%s]" % (f_name, self.root_dir))
        return file_path

    def create_header(self):
        f_path = self._find_file(self.names_file)

        #

