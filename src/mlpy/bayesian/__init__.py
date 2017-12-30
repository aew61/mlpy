# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
from nb import nb  # noqa: E402
import stdlabels  # noqa: E402
from abstractlabel import abstractlabel  # noqa: E402
from text_classifier import text_classifier  # noqa: E402

