# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
# from contok import contok

from cbow import cbow
from skipgram import skipgram
# from word2vec import word2vec

