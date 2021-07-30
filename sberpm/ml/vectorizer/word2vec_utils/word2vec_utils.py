from typing import Iterable, Tuple
from collections import defaultdict


class Word2VecException(Exception):
    pass


class Word2VecVocabulary:
    """Vocabulary for Word2Vec"""

    def __init__(self, tokenized_corpus: Iterable, min_word: int = 1):
        """
        Initializing vocabulary

        Parameters
        ----------
        tokenized_corpus : Iterable
            corpus like list of lists of tokens
        """
        self._vocabulary, self.word_count_dict = \
            Word2VecVocabulary._get_vocabulary(tokenized_corpus)
        self.token2idx = {token: idx for idx, token in
                          enumerate(self._vocabulary)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.size = len(self._vocabulary)

    @staticmethod
    def _get_vocabulary(tokenized_corpus: Iterable, min_word: int = 1) \
            -> Tuple[set, defaultdict]:
        """
        Get vocabulary method

        Parameters
        ----------
        tokenized_corpus : Iterable
            corpus like list of lists of tokens

        Returns
        -------
        vocabulary : set
            set of tokens
        """
        vocabulary = set()
        word_count_dict = defaultdict(int)
        for sentence in tokenized_corpus:
            for token in sentence:
                word_count_dict[token] += 1
                vocabulary.add(token)
        for token in word_count_dict.keys():
            if word_count_dict[token] < min_word:
                del word_count_dict[token]
                vocabulary.remove(token)
        return vocabulary, word_count_dict
