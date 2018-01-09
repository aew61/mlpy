# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
import features


def load_c45(names_file, data_file, root_dir):
    header = features.FHeader(names_file, root_dir=root_dir)
    header.create_header()
    dataset = features.Dataset(data_file, header, root_dir=root_dir)
    dataset._parse_data_file()
    dataset.delete_feature("index")

    return dataset.get_dataset()

