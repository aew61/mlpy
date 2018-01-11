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
from node import Node


def check_tree(t):
    assert(t.data == 0)
    assert(t.children[0].data == 1)
    assert(t.children[1].data == 2)
    assert(t.children[0].children[0].data == 3)
    assert(t.children[0].children[1].data == 4)
    assert(t.children[1].children[0].data == 5)
    assert(t.children[1].children[1].data == 6)

    print([n.data for n in t.interiors()])


def build_tree(l):
    n = Node(l[0])
    assert(n.data == l[0])

    ln1 = Node(l[1])
    rn1 = Node(l[2])
    assert(ln1.data == l[1])
    assert(rn1.data == l[2])

    ln2 = Node(l[3])
    rn2 = Node(l[4])
    assert(ln2.data == l[3])
    assert(rn2.data == l[4])

    ln3 = Node(l[5])
    rn3 = Node(l[6])
    assert(ln3.data == l[5])
    assert(rn3.data == l[6])

    n.append_child(ln1)
    n.append_child(rn1)

    ln1.append_child(ln2)
    ln1.append_child(rn2)

    rn1.append_child(ln3)
    rn1.append_child(rn3)

    return n


def main():
    l = [i for i in range(7)]  # 7 numbers
    check_tree(build_tree(l))


if __name__ == "__main__":
    main()

