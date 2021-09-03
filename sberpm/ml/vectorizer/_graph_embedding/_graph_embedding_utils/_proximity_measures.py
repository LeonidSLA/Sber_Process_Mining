from typing import Tuple

import numpy as np
from scipy.sparse.linalg import eigsh


def katz_index(a_m: np.ndarray, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
        Calculates the katz centrality of a graph

        Parameters
        ----------
        a_m: np.matrix
            Graph's adjacency matrix
        beta: float
            Decay parameter, default = -1. Should be less than the spectral
            radius of the adjacency matrix.

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
    """

    Parameters
    ----------
    a_m
    alpha

    Returns
    -------

    """
    assert (0 <= alpha < 1), (
        "Random walk probability should be between 0 and 1"
    )
    n = a_m.shape[1]
    d = np.diag(np.sum(a_m, axis=1))
    tr_matrix = d.inv() @ a_m
    idx = np.identity(n)
    m_l = (1 - alpha) * idx
    m_g = idx - alpha * tr_matrix
    return m_g, m_l


def common_neighbors(a_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    a_m

    Returns
    -------

    """
    return np.identity(a_m.shape[0]), np.linalg.matrix_power(a_m, 2)


def adamic_adar(a_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    a_m

    Returns
    -------

    """
    n = a_m.shape[0]
    d = np.diag(a_m)
    d = (np.sum(a_m, axis=1) + np.sum(a_m, axis=0)) - d
    d.diag()
    m_g = np.identity(n)
    m_l = a_m @ d @ a_m
    return m_g, m_l
