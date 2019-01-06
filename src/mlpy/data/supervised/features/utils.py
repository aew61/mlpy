# SYSTEM IMPORTS
import re
import os


# PYTHON PROJECT IMPORTS


__COMMENT_REGEX__ = "\|.*$|//.*$"
__WHITESPACE_REGEX__ = "\s+"


def trim(line):
    # remove multiple whitespaces
    line = re.sub(__WHITESPACE_REGEX__, " ", line).strip()
    # print(line)

    # remove comments
    line = re.sub(__COMMENT_REGEX__, "", line).strip()
    # print(line)

    # replace period
    if len(line) > 0 and line[-1] == ".":
        line = line[:-1].strip()
    return line

