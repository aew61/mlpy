# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
import node


class Tree(object):
    def __init__(self):
        self.root = None

    def topdown(self):
        if self.root is None:
            return None
        return self.root.topdown()

    def bottomup(self):
        if self.root is None:
            return None
        return self.root.bottomup()

    def leaves(self):
        if self.root is None:
            return None
        return self.root.leaves()

    def interiors(self):
        if self.root is None:
            return None
        return self.root.interiors()

