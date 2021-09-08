from typing import Callable, Tuple

import numpy as np
from scipy.sparse.linalg import svds

from sberpm.ml.vectorizer._graph_embedding._proximity_measures import katz_index


class GraphEmbedder:
    """
    Class for embedding directed graphs based on the algorithm from
    dl.acm.org/doi/pdf/10.1145/2939672.2939751
    Similarity between i-th source and j-th destination vectors indicates
    a directed edge from vertex i to vertex j.

    Parameters
    ----------
    adj_m: np.ndarray, default=None
        The adjacency matrix of a directed graph.

    Attributes
    ----------
    _adj_m: np.ndarray
        The adjacency matrix of the directed graph.
   """

    def __init__(self, adj_m=None):
        assert (adj_m is not None), (
            "Pass a valid adjacency matrix with nonnegative entries"
        )
        assert (adj_m.shape[0] == adj_m.shape[1]), (
            "Adjacency matrix should be square"
        )
        self._adj_m = adj_m

    def embed(self, k: int, prox_func: Callable = katz_index, *args) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Embedding function to embed GraphEmbedder internal adjacency matrix to
        dimension k.

        Parameters
        ----------
        k: int
            Dimension to reduce the adjacency matrix to.
        prox_func: Callable, default = katz_index
            Any proximity function that must return two matrix polynomials m_g
            and m_l whose modified product inv(m_g) @ m_l gives a high-order
            proximity matrix S such that S[i][j] indicates an edge between
            vertex i and vertex j.

            Five proximity functions (katz_index, rooted_pr, adamic_adar,
            common_neighbors, vertex_similarity) are implemented in
            sberpm.ml.vectorizer.

        *args:
            Rest of arguments needed for the proximity function.

        Returns
        -------
        [u_s, u_t]: Tuple[np.ndarray, np.ndarray]
            Row i of u_s (source) or u_t (target) is the embedding vector of
            vertex i.
            Similarity of vectors u_s[i] and u_t[j] for 0 <= i <= k and
            0 <= j <= k indicates a directed edge from vertex i to vertex j.

        """
        assert (0 < k < self._adj_m.shape[0]), (
            "Reduced dimension should between 0 and matrix dimensions"
        )
        m_g, m_l = prox_func(self._adj_m, *args)
        assert ((m_g is not None) and (m_l is not None)), (
            "Please use proximity function that returns two correct matrix"
            "polynomials "
        )
        m_g = np.linalg.inv(m_g)
        # casting to float for sparse matrix methods
        M = (m_g @ m_l).astype(float)
        u_s, s, u_t = svds(M, k, which="LM")
        u_s = u_s.transpose()
        n = u_t.shape[1]
        s = np.resize(np.sqrt(s), (k, n))
        u_s *= s
        u_t *= s
        return u_s, u_t
