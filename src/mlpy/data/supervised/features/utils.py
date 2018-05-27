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


def find_file(f_name, root_dir):
    # look for f_name in root_dir and subdirectories
    file_path = None
    if os.path.exists(f_name):
        file_path = f_name
    else:
        for dir_path, _, _ in os.walk(root_dir, topdown=True):
            if os.path.exists(os.path.join(dir_path, f_name)):
                return os.path.join(dir_path, f_name)

    if file_path is None:
        raise ValueError("Could not find file [%s] in root_dir [%s]" % (f_name, root_dir))
    return file_path

