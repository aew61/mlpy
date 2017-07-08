# SYSTEM IMPORTS
import os
import sys


_current_dir = os.path.abspath(os.path.dirname(__file__))
_scripts_dir = os.path.join(_current_dir, "scripts")
_dirs_to_add = [_current_dir, _scripts_dir]
for _dir in _dirs_to_add:
    if _dir not in sys.path:
        sys.path.append(_dir)
del _current_dir
del _scripts_dir
del _dirs_to_add


# PYTHON PROJECT IMPORTS
import activation_functions  # noqa: E402
import nets as nets  # noqa: E402
import qlearn  # noqa: E402
import processes  # noqa: E402
import bayesian  # noqa: E402
import data_partitioning  # noqa: E402

