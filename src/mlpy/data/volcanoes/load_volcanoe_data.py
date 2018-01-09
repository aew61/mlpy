# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_  not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
import features


def load_volcanoe_data():
    cd = os.path.abspath(os.path.dirname(__file__))
    names_file = "volcanoes"
    data_file = names_file

    header = features.FHeader(names_file, root_dir=cd)
    header.create_header()
    dataset = features.Dataset(data_file, header, root_dir=cd)
    dataset._parse_data_file()

    return header.get_header(), dataset.X, dataset.Y, dataset


