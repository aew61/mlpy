# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
from contok import contok


def load_file(file_path, tokenization_func=None, context_size):
    data = list()
    with open(file_path, "r") as f:
        for l in f:
            data.append(l)
    return data
