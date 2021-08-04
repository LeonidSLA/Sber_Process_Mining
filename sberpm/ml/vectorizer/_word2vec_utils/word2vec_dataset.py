from typing import List, Union
import re
import string

from torch.utils.data import Dataset
import torch
import pandas as pd

from .word2vec_utils import Word2VecVocabulary, Word2VecException

PAD_TOKEN_INDEX = 0


class Word2VecDataset(Dataset):
    """Inspired by github.com/goddoe"""

    def __init__(self,
                 corpus=None,
                 corpus_path: str = None,
                 pad_token: str = "<pad>",
                 pad_token_idx: int = PAD_TOKEN_INDEX,
                 tokenizer=None,
                 context_size: int = 2,
                 min_word: int = 1,
                 is_cbow: bool = True):
        """

        Parameters
        ----------
        corpus : iterable of sentences or str
        corpus_path : path to file
        tokenizer : tokenizer
        context_size : window size
        min_word : minimal word count for word to be in vocabulary
        """
        if corpus is None and corpus_path is None:
            raise Word2VecException(
                "corpus_path or corpus was expected"
            )
        if corpus_path:
            if corpus:
                print(f"Ignoring {corpus} and reading file from {corpus_path}")
            with open(corpus_path, "r") as file:
                self.corpus = file.read()
        else:
            self.corpus = corpus
        self.pad_token = pad_token
        self.pad_idx = PAD_TOKEN_INDEX
        if isinstance(self.corpus, str):
            self.tokenized_corpus = Word2VecDataset \
                ._tokenize_corpus(self.corpus, tokenizer=tokenizer)
        elif isinstance(self.corpus, List) \
                or isinstance(self.corpus, pd.Series):
            self.tokenized_corpus = corpus
        self.vocab = Word2VecVocabulary(self.tokenized_corpus,
                                        min_word=min_word,
                                        pad_token=pad_token,
                                        pad_token_idx=pad_token_idx)
        self.data = []
        max_len = max(map(len, self.tokenized_corpus))
        assert max_len >= 2 * context_size, (
            "There is no data after preprocessing, "
            "because 2 * context_size > max_len of sequence, "
            "please reduce context_size"
        )
        for line in self.tokenized_corpus:
            for i in range(context_size, len(line) - context_size):
                context = [self.vocab.token2idx[line[i + d]]
                           for d in range(-context_size, context_size + 1)
                           if d != 0]
                if is_cbow:
                    self.data += [(context, self.vocab.token2idx[line[i]])]
                else:
                    self.data += [(self.vocab.token2idx[line[i]], context)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.LongTensor(sample[0]),
            torch.LongTensor([sample[1]]),
        )

    @staticmethod
    def _preprocess_corpus(corpus: Union[List[str], str]) \
            -> Union[List[str], str]:
        if isinstance(corpus, str):
            return corpus \
                .lower() \
                .translate(str.maketrans('', '', string.punctuation))
        elif isinstance(corpus, List):
            return list(map(lambda x: x.lower()
                                       .translate(
                str.maketrans('', '', string.punctuation)),
                            corpus))
        else:
            raise TypeError(
                f"Expected corpus type from [str, List[str]], "
                f"but got {type(corpus)}"
            )

    @staticmethod
    def _tokenize_corpus(corpus, preprocess=True, tokenizer=None) \
            -> List:
        """
        Tokenizing corpus
        Parameters
        ----------
        corpus : iterable of sentences
        tokenizer : tokenizer

        Returns
        -------
        tokenized_corpus : List
            List of list with tokens
        """
        tokenized_corpus = []
        if preprocess:
            corpus = Word2VecDataset._preprocess_corpus(corpus)
        for line in re.split(r"\n+", corpus):
            if tokenizer is None:
                tokenized_corpus += [line.split()]
            else:
                tokenized_corpus += [tokenizer(line)]
        return tokenized_corpus
