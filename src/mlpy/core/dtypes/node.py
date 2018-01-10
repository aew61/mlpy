# SYSTEM IMPORTS
import os
import sys


# PYTHON PROJECT IMPORTS


class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = list()
        self.parent = None

    def topdown(self):
        yield self
        for c in self.children:
            for n in c.topdown():
                yield n

    def bottomup(self):
        for c in self.children:
            for n in c.bottomup():
                yield n
        yield self

    def leaves(self):
        if len(self.children) == 0:
            yield self
        else:
            for c in self.children:
                for l in c.leaves():
                    yield l

    def interiors(self):
        if len(self.children) > 0:
            yield self
            for c in self.children:
                for i in c.interiors():
                    yield i

    def append_child(self, node):
        node.parent = self
        self.children.append(node)

    def insert_child(self, node, i):
        node.parent = self
        self.children.insert(i, node)

    def delete_child(self, i):
        self.children[i].parent = None
        self.children[i:i+1] = []

