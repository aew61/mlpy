# SYSTEM IMPORTS


# PYTHON PROJECT IMPORTS


NOMINAL = 0
ORDERED = 1
CONTINUOUS = 2
HIERARCHICAL = 3


def is_discrete(f_type):
    return f_type == NOMINAL or f_type == ORDERED


def is_continuous(f_type):
    return f_type == CONTINUOUS


def is_hierarchical(f_type):
    return f_type == HIERARCHICAL


def has_order(f_type):
    return f_type == ORDERED or is_continuous(f_type) or is_hierarchical(f_type)

