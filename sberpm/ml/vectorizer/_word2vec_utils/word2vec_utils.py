from typing import Iterable, Tuple
from collections import defaultdict


class Word2VecException(Exception):
    pass


class Word2VecVocabulary:
    """Vocabulary for Word2Vec"""

    def __init__(self,
                 tokenized_corpus: Iterable,
                 min_word: int = 1,
                 pad_token: str = "<pad>",
                 pad_token_idx: int = 0):
        """
        Initializing vocabulary

        Parameters
        ----------
        tokenized_corpus : Iterable
            corpus like list of lists of tokens
        """
        self._vocabulary, self.word_count_dict = \
            Word2VecVocabulary._get_vocabulary(tokenized_corpus,
                                               min_word,
                                               pad_token)
        self.idx2token = {idx: token for idx, token
                          in enumerate(self._vocabulary, start=0)}
        if self.idx2token[pad_token_idx] != pad_token:
            not_pad_token_with_pad_token_idx = self.idx2token[pad_token_idx]
            tmp_pad_token_idx = None
            for idx, token in self.idx2token.items():
                if token == pad_token:
                    tmp_pad_token_idx = idx
                    break
            if tmp_pad_token_idx is None:
                raise Word2VecException(
                    "There is no pad_token in the vocabulary"
                )
            self.idx2token[tmp_pad_token_idx] = not_pad_token_with_pad_token_idx
            self.idx2token[pad_token_idx] = pad_token
        self.token2idx = {token: idx for idx, token in self.idx2token.items()}
        self.token2idx = dict(sorted(self.token2idx.items(),
                                     key=lambda item: item[1]))
        self.size = len(self.token2idx)

    @staticmethod
    def _get_vocabulary(tokenized_corpus: Iterable,
                        min_word: int = 1,
                        pad_token: str = "<pad>") \
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
        vocabulary = {pad_token}
        word_count_dict = defaultdict(int)
        word_count_dict[pad_token] = 1
        for sentence in tokenized_corpus:
            for token in sentence:
                word_count_dict[token] += 1
                vocabulary.add(token)
        for token in word_count_dict.keys():
            if word_count_dict[token] < min_word and token != pad_token:
                del word_count_dict[token]
                vocabulary.remove(token)
        return vocabulary, word_count_dict
