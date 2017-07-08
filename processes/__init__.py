# SYSTEM IMPORTS
import os
import sys


_current_dir_ = os.path.abspath(os.path.dirname(__file__))
if _current_dir_ not in sys.path:
    sys.path.append(_current_dir_)
del _current_dir_


# PYTHON PROJECT IMPORTS
from hmm import hmm

