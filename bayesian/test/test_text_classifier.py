# SYSTEM IMPORTS
import os
import re
import sys


_current_dir_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_current_dir_, "..")
_dirs_to_add_ = [_current_dir_, _src_dir_]
for _dir_ in _dirs_to_add_:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _dirs_to_add_
del _src_dir_
del _current_dir_


# PYTHON PROJECT IMPORTS
from text_classifier import TextClassifier
from stdlabels import discrete_label


if __name__ == "__main__":
    news_set = ["not to eat too much is not enough to lose weight",
                "Russia is trying to invade Ukraine",
                "do not neglect exercise",
                "Syria is the main issue, Obama says",
                "eat to lose weight",
                "you should not eat much"]
    news_annotations = ["health", "politics", "health", "politics", "health", "health"]

    class_label = discrete_label("topic", ["health", "politics"])
    tokenization_string = "\"'().,!-<>/\\"

    def word_filter(example):
        return [x for x in re.sub(str([tokenization_string]), "", example.lower()).split() if len(x) > 3]

    classifier = TextClassifier()
    classifier.train(news_set, news_annotations,
                     class_label, word_filter)

    unknown_instance = "Even if I eat too much, it is not possible to lose some weight"
    print("classifications for %s: %s" % (class_label.get_all_states(),
                                          classifier.classify_text(unknown_instance)))

