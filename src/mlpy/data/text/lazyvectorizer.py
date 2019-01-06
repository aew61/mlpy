# SYSTEM IMPORTS


# PYTHON PROJECT IMPORTS


class lazyvectorizer(object):
    def __init__(self, vocab, tokenized_text):
        self.vocab = vocab
        self._vocab_dict = {w: i for i, w in enumerate(self.vocab)}
        self.tokenized_text = tokenized_text


