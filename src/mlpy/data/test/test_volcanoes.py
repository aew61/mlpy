# SYSTEM IMPORTS
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_cd_, "..")
for _dir_ in [_cd_, _src_dir_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import features


def main():
    cd = os.path.abspath(os.path.dirname(__file__))
    volcanoes_dir = os.path.join(cd, "..", "volcanoes")
    names_file = "volcanoes"
    data_file = "volcanoes"

    print(volcanoes_dir)

    fheader = features.FHeader(names_file, root_dir=volcanoes_dir)
    fheader.create_header()
    print(fheader.feature_names)
    print(fheader.get_header())

    dataset = features.Dataset(data_file, fheader, root_dir=volcanoes_dir)
    dataset._parse_data_file()

    print(dataset.X.shape)
    print(dataset.Y.shape)


if __name__ == "__main__":
    main()

