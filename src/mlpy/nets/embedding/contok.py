# SYSTEM IMPORTS
import numpy
import os
import sys


_src_dir_ = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")
if _src_dir_ not in sys.path:
    sys.path.append(_src_dir_)
del _src_dir_


# PYTHON PROJECT IMPORTS
import symbols


class contok(object):
    def __init__(self, context_size, reserved_symbols=None):
        self.context_size = context_size
        self.vocab = set()
        self.vocab_size = 0
        self.token_map = dict()
        self.index_map = dict()
        if reserved_symbols is None:
            reserved_symbols = set([symbols.UNKNOWN_SYMBOL, symbols.START_SYMBOL, symbols.END_SYMBOL])
        self.reserved_symbols = reserved_symbols
        

    def tokenize(self, corpus, tokenization_func=None, update_existing=False):
        if tokenization_func is None:
            tokenization_func = lambda text: text.strip().rstrip().split()

        if not update_existing:
            self.vocab = set(self.reserved_symbols)

        for l in corpus:
            self.vocab.update(tokenization_func(l))
        self.vocab_size = len(self.vocab)
        for i, t in enumerate(self.vocab):
            self.token_map[t] = i
            self.index_map[i] = t
        return self

    def transform(self, corpus, tokenization_func=None):
        token_contexts = list()
        target_tokens = list()

        if tokenization_func is None:
            tokenization_func = lambda text: text.strip().rstrip().split()

        for l in corpus:
            tok_l = ([symbols.START_SYMBOL] * self.context_size) + tokenization_func(l) +\
                ([symbols.END_SYMBOL] * self.context_size)
            for i in range(self.context_size, len(tok_l) - self.context_size):
                token_contexts.append(tok_l[i-self.context_size:i])
                target_tokens.append(tok_l[i])
                token_contexts.append(tok_l[i+1:i+1+self.context_size])
                target_tokens.append(tok_l[i])

        # for l, o in zip(token_contexts, target_tokens):
        #     print("%s -> %s" % (l, o))

        # check data
        for i, (l, o) in enumerate(zip(token_contexts, target_tokens)):
            assert(len(l) == self.context_size)
            one_hot_context = numpy.zeros(self.context_size * self.vocab_size)
            for j, t in enumerate(l):
                one_hot_context[j*self.vocab_size + self.token_map[t]] = 1
            one_hot_target = numpy.zeros(self.vocab_size)
            one_hot_target[self.token_map[o]] = 1
            token_contexts[i] = one_hot_context
            target_tokens[i] = one_hot_target

        return numpy.array(token_contexts), numpy.array(target_tokens)

