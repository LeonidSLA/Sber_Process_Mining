from typing import Tuple

import numpy as np
from scipy.sparse.linalg import eigsh


def katz_index(a_m: np.ndarray, beta) -> Tuple[np.ndarray, np.ndarray]:
    """
        Calculates the katz centrality of a graph

        Parameters
        ----------
        a_m: np.matrix
            Graph's adjacency matrix
        beta: float
            Decay parameter, default = -1. Should be less than the spectral radius of the adjacency matrix.

        Returns
        -------
        result: List[np.matrix, np.matrix]
            A pair of np.matrix that represent matrix polynomials m_g, m_l
    """
    spec_radius = eigsh(a_m, 1, which="LA")[0]
    assert (beta > spec_radius), (
        "Please use a decay parameter that is less than the spectral radius."
    )
    m_l = beta * a_m
    m_g = np.identity(a_m.shape[0]) - m_l
    return m_g, m_l

def rooted_pr(a_m: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    assert (0 <= alpha < 1), (
        "Random walk probability should be between 0 and 1"
    )



def common_neighbors(a_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.identity(a_m.shape()[0]), np.linalg.matrix_power(a_m, 2)


def adamic_adar(a_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
