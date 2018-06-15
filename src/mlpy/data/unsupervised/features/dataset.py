# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS


__DATA_FILE_EXTENSION__ = ".data"

class Dataset(object):
    def __init__(self, data_file, fheader_obj, root_dir="."):
        self.fheader_obj = fheader_obj

        self.num_features = 0
        self.data_file = data_file

        self.X = None

        if not self.data_file.endswith(__DATA_FILE_EXTENSION__):
            self.data_file += __DATA_FILE_EXTENSION__
        self.root_dir = root_dir

    def _parse_data_file(self):
        data_fpath = utils.find_file(self.data_file, root_dir=self.root_dir)
        # data = loaders.

