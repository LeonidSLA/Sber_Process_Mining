from typing import List

from torch.utils.data import Dataset

from .word2vec_utils import Word2VecVocabulary, Word2VecException


class Word2VecDataset(Dataset):
    """Inspired by github.com/goddoe"""
    def __init__(self,
                 corpus=None,
                 corpus_path: str = None,
                 tokenizer=None,
                 context_size: int = 2,
                 min_word: int = 1):
        """

        Parameters
        ----------
        corpus : iterable of sentences
        corpus_path : path to file
        tokenizer : tokenizer
        context_size : window size
        min_word : minimal word count for word to be in vocabulary
        """
        if corpus is None and corpus_path is None:
            raise Word2VecException(
                "corpus_path or corpus was expected"
            )
        self.corpus = corpus
        if corpus_path:
            if corpus:
                print(f"Ignoring {corpus} and reading file from {corpus_path}")
            with open(corpus_path, "r") as file:
                self.corpus = file.read()
        self.vocab, self.word_count_dict = \
            Word2VecVocabulary(Word2VecDataset
                               ._tokenize_corpus(self.corpus, tokenizer),
                               min_word=min_word)
        self.data = []
        for line in self.corpus:
            for i in range(context_size, len(line) - context_size):
                context = [self.vocab.token2idx[line[i + d]]
                           for d in range(-context_size, context_size + 1)
                           if d != 0]
                self.data.append((context, self.vocab.token2idx[line[i]]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def _tokenize_corpus(corpus, tokenizer=None) -> List:
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
        for line in corpus:
            if tokenizer is None:
                tokenized_corpus += [line.split()]
            else:
                tokenized_corpus += [tokenizer(line)]
        return tokenized_corpus
