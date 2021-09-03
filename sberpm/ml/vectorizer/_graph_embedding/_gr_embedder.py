from typing import Callable, Tuple

import numpy as np
from scipy import sparse

from _graph_embedding_utils._proximity_measures import katz_index


class GraphEmbedder:
    """
    Class for embedding directed graphs based on the algorithm from dl.acm.org/doi/pdf/10.1145/2939672.2939751
    Similarity between i-th source and j-th destination vectors indicates a directed edge from vertex i to vertex j.

    Parameters
    ----------
    adj_m: np.ndarray
        The adjacency matrix of the directed graph.

    Attributes
    ----------
    _adj_m: np.ndarray
        The adjacency matrix of the directed graph.
   """

    def __init__(self, adj_m=None):
        if adj_m is None:
            # do something to get an adjacency matrix
            # e.g. call a miner
            pass
        assert (adj_m.shape[0] == adj_m.shape[1]), (
            "Adjacency matrix should be square"
        )
        self._adj_m = adj_m

    def embed(self, k: int, prox_func: Callable = katz_index, *args) -> Tuple[np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        k
        prox_func

        Returns
        -------

        """
        assert (k < self._adj_m.shape[0]), (
            "Reduced dimension should be lower than the matrix dimension"
        )
        m_g, m_l = prox_func(self._adj_m, args)
        assert ((m_g is not None) and (m_l is not None)), (
            "Please use proximity function that returns two correct matrix"
            "polynomials "
        )
        m_g = np.linalg.inv(m_g)
        v_s, s, v_t = sparse.linalg.svd(m_g @ m_l, k, which='LM')
        v_s.transpose()
        m = v_s.shape[1]
        n = v_t.shape[1]
        s.sqrt().transpose()
        v_s *= np.resize(s, (n, k))
        v_t *= np.resize(s, (m, k))
        u_s = v_s[:k, :k]
        u_t = v_s[:k, :k]
        return u_s, u_t
