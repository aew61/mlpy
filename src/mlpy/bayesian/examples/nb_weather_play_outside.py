# SYSTEM IMPORTS
import matplotlib.pyplot as plt
import numpy
import os
import sys


_current_dir_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_current_dir_, "..", "..")
_dirs_to_add_ = [_current_dir_, _src_dir_]
for _dir_ in _dirs_to_add_:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _dirs_to_add_
del _src_dir_
del _current_dir_


# PYTHON PROJECT IMPORTS
from bayesian import nb
from bayesian import stdlabels


def main():
    weather_label = stdlabels.discrete_label("weather", ["sunny", "overcast", "rainy"], default_state="rainy")
    play_label = stdlabels.boolean_label("play outside")

    classifier = nb([weather_label], play_label)

    dataset = [
        {"weather": "sunny"},
        {"weather": "overcast"},
        {"weather": "rainy"},
        {"weather": "sunny"},
        {"weather": "sunny"},
        {"weather": "overcast"},
        {"weather": "rainy"},
        {"weather": "rainy"},
        {"weather": "sunny"},
        {"weather": "rainy"},
        {"weather": "sunny"},
        {"weather": "overcast"},
        {"weather": "overcast"},
        {"weather": "rainy"},
    ]

    annotations = [
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        True,
        True,
        False,
    ]

    assert(len(dataset) == len(annotations))

    classifier.train(dataset, annotations)
    feature_vector = {"weather": "sunny"}
    distribution = classifier.classify_pr(feature_vector)
    print(("play outside when its sunny distribution (%s): %s\n" +
           "play outside when its sunny scores       (%s): %s") %
           (play_label.get_all_states(), distribution,
            play_label.get_all_states(), classifier.classify_scores(feature_vector)))


if __name__ == "__main__":
    main()

