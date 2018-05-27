# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_cd_, "..")
for _dir_ in [_cd_, _src_dir_]:
    if _dir_  not in sys.path:
        sys.path.append(_dir_)
del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import loaders


def load_volcanoe_data():
    cd = os.path.abspath(os.path.dirname(__file__))
    names_file = "volcanoes"
    data_file = names_file

    return loaders.c45.load_c45(names_file, data_file, cd)

