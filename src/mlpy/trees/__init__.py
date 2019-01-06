# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_

# PYTHON PROJECT IMPORTS
from id3 import id3tree
from leid3 import leid3tree
from ppyid3 import ppyid3tree

