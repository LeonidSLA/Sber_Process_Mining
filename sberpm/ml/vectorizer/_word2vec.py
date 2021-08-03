from collections import defaultdict
from typing import List, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import torch

from .word2vec_utils.word2vec_dataset import Word2VecDataset
from .word2vec_utils.model import CBOWModel
from .word2vec_utils.train import train

DEFAULT_OPTIMIZER = torch.optim.Adam
DEFAULT_LR = 1e-3
DEFAULT_N_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128


class TextWord2Vec:
    """
    Class for vectorizing textual data with Word2Vec algorithm.

    Parameters
    ----------
    size: int, default=100
        Dimension of the resulting vectors.

    Attributes
    ----------
    _word2vec: dict of {str: list of float}
        Key: a token (a word), value: its vector representation.

    _dim: int
        Size of the vocabulary.

    _size: int
        Dimension of the resulting vectors.

    Examples
    --------
    >>> from sberpm.ml.vectorizer import TextWord2Vec
    >>> from sberpm.ml.text import TextPreprocessing
    >>> preprocessing = TextPreprocessing()
    >>> vectorizer = TextWord2Vec(size = 100)
    >>> clear_text = preprocessing.transform(text)
    >>> embeddings = vectorizer.fit(clear_text).transform(clear_text)
    """

    def __init__(self, size=100):
        self._word2vec = None
        self.word2weight = None
        self._dim = None
        self._size = size
        self._model = None
        self._nn = None
        self._vocab = None

    def fit(self, text_data: Union[str, List[List[str]]]):
        """
        Fit Word2Vec

        Parameters
        ----------
        text_data : list of list of str or str

        Returns
        -------
        self :
            TextWord2Vec
        """
        w2v_dataset = Word2VecDataset(corpus=text_data)
        self._vocab = w2v_dataset.vocab
        model = CBOWModel(w2v_dataset.vocab.size, self._size)
        optimizer = DEFAULT_OPTIMIZER
        lr = DEFAULT_LR
        _ = train(w2v_dataset,
                  model,
                  n_epoch=DEFAULT_N_EPOCHS,
                  verbose=False,
                  batch_size=DEFAULT_BATCH_SIZE,
                  optimizer_=optimizer,
                  optimizer_params={"lr": lr})
        self._model = model
        self._word2vec = dict(zip(w2v_dataset.vocab.token2idx.keys(),
                                  model.get_embeddings()))
        self._dim = len(self._word2vec.values())
        return self

    def transform(self,
                  text_data: List[List[str]]) -> np.array:
        """
        Gets embeddings of the textual data using trained Word2Vec algorithm.

        Parameters
        ----------
        text_data : list of list of str
            List of of textual documents
                that are represented by a list of tokens.

        Returns
        -------
        embeddings: np.array of float32
            List of vectorized documents.
        """
        return np.array([
            np.mean([self._word2vec[w] for w in document if
                     w in self._word2vec] or [np.zeros(self._dim)], axis=0)
            for document in text_data
        ])

    def most_similar(self,
                     target: str,
                     n: int = 5) -> List:
        """
        Finds the top-N most similar words to the given one.

        This method computes cosine similarity between
        a simple mean of the projection weight vectors
        of the given words and the vectors for each word in the model.

        Parameters
        ----------
        target: str
            Similar words will be found to the target word.

        n: int, default=5
            Number of similar words to find.

        Returns
        -------
        result: List[Union[str, float]]
            List of tuples: a similar word and its distance to the given one.
        """
        assert self._model is not None, (
            "Please use fit() before most_similar()"
        )
        if self._nn is None:
            nn = NearestNeighbors()
            nn.fit(self._model.get_embeddings())
            self._nn = nn
        target_embd = self._word2vec[target]
        distances, similar_indices = self._nn.kneighbors([target_embd],
                                                         n_neighbors=n + 1,
                                                         return_distance=False)
        distances = distances[0]
        similar_indices = similar_indices[0]
        similar_words = [self._vocab.idx2token[idx] for idx in similar_indices]
        return list(zip(similar_words, distances))

    def transform_weighted(self,
                           text_data: List[List[str]]):
        """
        Gets embeddings of the textual data weighted by TF-IDF
        using trained Word2Vec algorithm.

        Parameters
        ----------
        text_data : list of list of str
            List of of textual documents that are represented
            by a list of tokens.

        Returns
        -------
        embeddings: np.array of float
            List of vectorized documents.
        """
        tf_idf = TfidfVectorizer(analyzer=lambda x: x)
        tf_idf.fit(text_data)
        max_idf = max(tf_idf.idf_)
        self.word2weight = defaultdict(lambda: max_idf,
                                       [(w, tf_idf.idf_[i]) for w, i in
                                        tf_idf.vocabulary_.items()])

        return np.array([
            np.mean(
                [self._word2vec[w] * self.word2weight[w] for w in document if
                 w in self._word2vec] or
                [np.zeros(self._dim)], axis=0)
            for document in text_data
        ])
