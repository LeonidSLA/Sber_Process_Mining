from pymorphy2 import MorphAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from numpy.linalg import svd
from sklearn.feature_extraction.text import CountVectorizer


class LSASummarizer:
    """
    Implementation of topic based Latent Semantic Analysis summarizer.

    Examples
    _________
    >>>from sberpm.ml.summarization import LSASummarizer
    >>>sumr=LSASummarizer()
    >>>sumr(texts)
    """

    def __call__(self, texts, n_sentences=10):
        """
        Implementation of topic based Latent Semantic Analysis summarizer.

        Parameters
        ----------
        texts: pd.Series or list of str
            List of texts for summarization.

        n_sentences: int, default=10
            Number of sentences in an abstract.

        Returns
        -------
        result: str
        """
        text = ''.join(texts)
        term_matrix = self._create_matrix(text)
        ranks = self._perform_ranking(term_matrix, n_sentences)
        sentences = sent_tokenize(text)

        ranks_sorted = sorted(((i, ranks[i], s) for i, s in enumerate(sentences)), key=lambda x: ranks[x[0]],
                              reverse=True)
        top_n = sorted(ranks_sorted[:n_sentences])

        return ' || '.join(x[2] for x in top_n)

    @staticmethod
    def _create_matrix(text):
        """
        Calculates term-document matrix.

        Parameters
        ----------
        term_dict: dict
            Dictionary that contains unique words(terms) of texts.

        text_cleaned_normalized: list of str

        Returns
        -------
        matrix: np.ndarray
        """
        lemmatizer = MorphAnalyzer()

        text_cleaned_normalized = list(
            map(lambda x: ' '.join([lemmatizer.parse(word)[0].normal_form for word in list(word_tokenize(x))]),
                sent_tokenize(text)))
        bow = CountVectorizer()

        return bow.fit_transform(text_cleaned_normalized).toarray().T

    @staticmethod
    def _perform_ranking(matrix, n):
        """
        Calculates ranks of each sentence in term-document matrix.

        Parameters
        ----------
        matrix: np.ndarray
        n: int

        Returns
        -------
        ranks: list of float
        """
        u_m, sigma, v_m = svd(matrix, full_matrices=False)
        powered_sigma = tuple(s ** 2 if i < n else 0.0 for i, s in enumerate(sigma))
        ranks = []

        for column_vector in v_m.T:
            rank = sum(s * v ** 2 for s, v in zip(powered_sigma, column_vector))
            ranks.append(np.sqrt(rank))

        return ranks
