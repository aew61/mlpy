# SYSTEM IMPORTS
import re


# PYTHON PROJECT IMPORTS


__COMMENT_REGEX__ = "\|.*$|//.*$"
__WHITESPACE_REGEX__ = "\s+"


def trim_line(line):
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


def c45_format(data):
    trimmed_data = list()
    for l in data:
        t_l = trim_line(l)
        if len(t_l) > 0:
            trimmed_data.append(t_l)
    return trimmed_data

    # return [t_l for t_l in [trim_line(l) for l in data] if len(t_l) > 0]

