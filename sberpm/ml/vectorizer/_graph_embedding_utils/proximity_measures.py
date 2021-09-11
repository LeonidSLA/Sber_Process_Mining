from typing import Tuple

import numpy as np
from scipy.sparse.linalg import eigs


def katz_index(a_m: np.ndarray, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates a proximity matrix based on each vertex katz centrality.

    Parameters
    ----------
    a_m: np.ndarray
        Graph's adjacency matrix
    beta: float
        Decay parameter. Must be less than the spectral
        radius of the adjacency matrix.

    Returns
    -------
    result: Tuple[np.ndarray, np.ndarray]
        A pair of np.ndarray that represent matrix polynomials m_g, m_l.
        (the inverse of m_g) @ m_l = vertex proximity matrix
    """
    # casting to float for sparse matrix eigenvalue search
    A = a_m.astype(float)
    spec_radius, _ = eigs(A, 1, which="LM")
    spec_radius = spec_radius.item(0)
    assert (np.abs(spec_radius) > np.abs(beta)), (
        "Please use a decay parameter that is less than the spectral radius."
    )
    m_l = beta * a_m
    m_g = np.identity(a_m.shape[0]) - m_l
    return m_g, m_l


def rooted_pr(a_m: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rooted PageRank algorithm for computing a proximity matrix
    between vertices.

    Parameters
    ----------
    a_m: np.ndarray
        Graph adjacency matrix
    alpha: float
        Random walk probability

    Returns
    -------
    result: Tuple[np.ndarray, np.ndarray]
        Pair of matrix polynomials m_g, m_l that give a
        proximity matrix when multiplied.
    """
    assert (0 <= alpha < 1), (
        "Random walk probability should be between 0 and 1"
    )
    n = a_m.shape[1]
    d = np.sum(a_m, axis=1)
    d = np.diag(d)
    tr_matrix = np.linalg.inv(d) @ a_m
    idx = np.identity(n)
    m_l = (1 - alpha) * idx
    m_g = idx - alpha * tr_matrix
    return m_g, m_l


def common_neighbors(a_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a vertices' proximity matrix based on common_neighbors approach.

    Parameters
    ----------
    a_m: np.ndarray
        Graph adjacency matrix.

    Returns
    -------
    result: Tuple[np.ndarray, np.ndarray]
        Pair of matrix polynomials m_g, m_l that give a
            proximity matrix when multiplied.
    """
    return np.identity(a_m.shape[0]), np.linalg.matrix_power(a_m, 2)


def adamic_adar(a_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adamic-adar based calculation of graph proximity matrix.

    Parameters
    ----------
    a_m: np.ndarray

    Returns
    -------
    result: Tuple[np.ndarray, np.ndarray]
        Pair of matrix polynomials m_g, m_l that give a
            proximity matrix when multiplied.
    """
    n = a_m.shape[0]
    d = np.diag(a_m)
    d = (np.sum(a_m, axis=1) + np.sum(a_m, axis=0)) - d
    d = np.diag(d)
    m_g = np.identity(n)
    m_l = a_m @ d @ a_m
    return m_g, m_l


def vertex_similarity(a_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gives a vertex proximity matrix based on paper doi 10.1137/S0036144502415960

    Parameters
    ----------
    a_m: np.ndarray
        Graph adjacency matrix.

    Returns
    -------
    result: Tuple[np.ndarray, np.ndarray]
        Pair of matrix polynomials m_g, m_l that give a
            proximity matrix when multiplied.
    """
    z = np.identity(a_m.shape[0])
    a_m_t = np.transpose(a_m)
    for i in range(24):
        z_new = a_m @ z @ a_m_t + a_m_t @ z @ a_m
        z = z_new / np.linalg.norm(z_new)
    return np.identity(a_m.shape[0]), z
