# SYSTEM IMPORTS
import numpy
from operator import itemgetter


# PYTHON PROJECT IMPORTS
from abstractlabel import abstractlabel


class nbnode(object):
    def __init__(self, label, dependent_labels=[], smoothing_coeff=0.0):
        self._label = label
        assert(isinstance(label, abstractlabel))

        # we have to store a truth tabel (of conditional probabilities) for this node.
        # For a simple example, this table will look like:
        # +-----+-----------------+-----------------+
        # | Foo |     label=T     |     label=F     |
        # +-----+-----------------+-----------------+
        # |  T  | Pr(l=T | Foo=T) | Pr(l=F | Foo=T) |
        # |  F  | Pr(l=T | Foo=F) | Pr(l=F | Foo=F) |
        # +-----+-----------------+-----------------+

        # but for full generality, it will look like:
        # +------+------+-----+------+----------------+----------------+-----+----------------+
        # | dep1 | dep2 | ... | depM |      l=s1      |      l=s2      | ... |      l=sN      |
        # +-----------------------------------------------------------------------------------+
        # |  s11 |  s12 | ... |  s1M | Pr(l=s1 | ...) | Pr(l=s2 | ...) | ... | Pr(l=sN | ...) |
        # |  s21 |  s22 | ... |  s2M | Pr(l=s1 | ...) | Pr(l=s2 | ...) | ... | Pr(l=sN | ...) |
        # |  ... |  ... | ... |  ... |      ...       |      ...       | ... |      ...       |
        # +------+------+-----+------+----------------+----------------+-----+----------------+

        # to make this table, we need to set a few things clear first:
        # 1) We need to sort our 'dependencies' so that we can always know which column
        #    to refer to given a dependency.
        #    I recommend that we also sort this list by the number of states for each dependency
        #    so that the truth table will 'look prettier,' idk I just kind of like that idea.
        # 2) We need to know the dimensions of our table. Given that each dependency
        #    d_i (0 <= i <= M) has k_i potential states, we need to figure out how many
        #    rows this table will have. If all the inputs are boolean functions, then the number
        #    of rows in the table is the same amount of boolean functions that we can represent
        #    with the number of inputs (2^l for l boolean inputs). However, how does this scale
        #    when there are multiple inputs that can have whatever number of states?
        #    Lets look at this theoretically:
        #        As we start from the last dependency column (smallest number of states),
        #        the number of rows = d_m
        #        Then, for each state in d_(m-1), there are d_m number of states,
        #        therefore there are d_(m-1) * d_m number of rows (so far).
        #        Repeating this for each dependency, we can see that the number of rows in our
        #        table is equal to the product of number of states for all dependencies i.e:
        #            rows = d_1 * d_2 * ... * d_(m-1) * d_m
        #    and the number of columns is the number of states that our label can have
        #    + the number of dependencies.

        # An optimization that we can use is to only store the conditional probabilities
        # and not the dependency states that were used in that row. We can instead use
        # the dependency states as the 'index' of the row, and save ourselves some memory (and time)

        # first we want to sort the list by the 'name' of the label
        self._dependent_labels = sorted(dependent_labels,
                                        key=lambda dependency: dependency.get_label())

        # second, sort by the number of states each dependency has, largest first
        self._dependent_labels = sorted(self._dependent_labels,
                                        key=lambda dependency: dependency.num_states(),
                                        reverse=True)
        self._rows, self._columns = self._compute_table_size()
        self._truth_table = numpy.full((self._rows, self._columns), smoothing_coeff, dtype=float)
        self._parents = [dep.get_label() for dep in self._dependent_labels]

    def _compute_table_size(self):
        rows = 1
        columns = 1 if self._label.num_states() is numpy.inf\
                    else self._label.num_states()
        for dependency in self._dependent_labels:
            rows *= 1 if dependency.num_states() is numpy.inf else dependency.num_states()
        return rows, columns

    def index_row(self, dependency_states_dict):
        index = 0

        # every time we advance a column (backwards), we need to offset the hash of that
        # dependency's value. This offset is the product of the number of states
        # that each dependency we have already accounted for has i.e.
        #   b_i = d_(i+1) * d_(i+2) ... d_(m-1) * d_m (where 0 <= i < m)
        #   and b_m = 1
        base = 1

        # iterate backwards over the labels
        for label in self._dependent_labels[::-1]:  # <list>[::-1] creates a shallow copy
            assert(label.get_label() in dependency_states_dict)
            index += (label.hash(dependency_states_dict[label.get_label()]) * base)
            base *= label.num_states()
        return index

    def conditional_pr(self, dependency_states_dict, label_state, default_pr=0.0):
        row_index = self.index_row(dependency_states_dict)
        col_index = self._label.hash(label_state)
        return self._truth_table[row_index][col_index] if row_index >= 0 and col_index >= 0\
                                                       else default_pr

