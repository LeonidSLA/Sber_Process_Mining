from collections import defaultdict
from typing import List, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import torch

from ._word2vec_utils.word2vec_dataset import Word2VecDataset
from ._word2vec_utils.model import CBOWModel
from ._word2vec_utils.train import train
from ._word2vec_utils.word2vec_utils import Word2VecException

DEFAULT_OPTIMIZER = torch.optim.Adam
DEFAULT_LR = 1e-3
DEFAULT_N_EPOCHS = 25
DEFAULT_BATCH_SIZE = 128
DEFAULT_SEED = 12345


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
        torch.random.manual_seed(DEFAULT_SEED)
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
        try:
            target_embd = self._word2vec[target]
        except KeyError:
            raise Word2VecException(
                f"There is no {target} in vocabulary"
            )
        distances, similar_indices = self._nn.kneighbors([target_embd],
                                                         n_neighbors=n + 1,
                                                         return_distance=True)
        distances = distances[0]
        similar_indices = similar_indices[0]
        similar_words = [self._vocab.idx2token[idx] for idx in similar_indices]
        return list(zip(similar_words, distances))[1:]

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


class ProcessWord2Vec:
    """
    Class for vectorizing event traces using Word2Vec algorithm.

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
    >>> import pandas as pd
    >>> from sberpm import DataHolder
    >>> from sberpm.ml.vectorizer import ProcessWord2Vec
    >>> # Create data_holder
    >>> df = pd.DataFrame({
    ...     'id_column': [1, 1, 2, 2],
    ...     'activity_column':['st1', 'st2', 'st1','st3'],
    ...     'dt_column':[123456, 123457, 123458,123459]})
    >>> data_holder = DataHolder(df,
    >>>                          'id_column', 'activity_column', 'dt_column')
    >>> vectorizer = ProcessWord2Vec(size = 2)
    >>> embeddings = vectorizer.fit(data_holder).transform(data_holder)
    """

    def __init__(self, size=10):
        self._word2vec = None
        self._dim = None
        self._vocab = None
        self._model = None
        self._size = size

    @staticmethod
    def _get_event_traces(data_holder):
        """
        Returns the event traces from the event log.

        Parameters
        ----------
        data_holder : DataHolder
            Object that contains the event log
            and the names of its necessary columns.

        Returns
        -------
        grouped_data: pd.Series of list of str
            List of event traces. An event trace is a list of str (activities).
        """
        return data_holder.get_grouped_columns(data_holder.activity_column)

    def fit(self, data_holder):
        """
        Fits Word2Vec vectorizer using the event traces in given data_holder.

        Parameters
        ----------
        data_holder : DataHolder
            Object that contains the event log
            and the names of its necessary columns.

        Returns
        -------
        self
        """
        event_traces = self._get_event_traces(data_holder)

        w2v_dataset = Word2VecDataset(corpus=event_traces)
        self._vocab = w2v_dataset.vocab
        torch.random.manual_seed(DEFAULT_SEED)
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

    def transform(self, data_holder):
        """
        Gets embeddings of the event traces in given data_holder using
        trained Word2Vec algorithm.

        Parameters
        ----------
        data_holder : DataHolder
            Object that contains the event log
            and the names of its necessary columns.

        Returns
        -------
        embeddings: np.array of float
            List of vectorized event traces.
        """
        event_traces = self._get_event_traces(data_holder)
        return np.array([
            np.mean([self._word2vec[w] for w in trace if w in self._word2vec]
                    or [np.zeros(self._dim)], axis=0)
            for trace in event_traces
        ])

    def transform_weighted(self, data_holder):
        """
        Gets embeddings of the event traces in given data_holder
        weighted by time duration of each activity
        using trained Word2Vec algorithm.

        Parameters
        ----------
        data_holder : DataHolder
            Object that contains the event log
            and the names of its necessary columns.

        Returns
        -------
        embeddings: np.array of float
            List of vectorized event traces.
        """
        event_traces = self._get_event_traces(data_holder)
        data_holder.check_or_calc_duration()
        duration_col = data_holder.data[data_holder.duration_column]
        data_holder.data['duration_norm'] = \
            (duration_col - duration_col.min()) \
            / (duration_col.max() - duration_col.min())
        dur = data_holder.get_grouped_columns('duration_norm')
        data_holder.data.drop('duration_norm', axis=1, inplace=True)
        embeddings = []
        for i in range(len(event_traces)):
            s_w = []
            s_t = 0
            for j in range(len(event_traces[i])):
                if event_traces[i][j] in self._word2vec:
                    s_w.append(self._word2vec[event_traces[i][j]] * dur[i][j])
                else:
                    s_w.append(0)
                s_t += dur[i][j]

            print(f"np.array(s_w): {np.array(s_w)}")
            print(f"s_t: {s_t}")
            embeddings.append(np.mean(np.array(s_w) / s_t, axis=0))

        return embeddings
