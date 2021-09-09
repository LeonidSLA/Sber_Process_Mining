from typing import Callable, Tuple, Dict

import numpy as np
from scipy.sparse.linalg import svds
from ._graph_embedding_utils.proximity_measures import katz_index
from sberpm.miners import SimpleMiner


class GraphEmbedder:
    """
    Class for embedding directed graphs based on the algorithm from
    doi 10.1145/2939672.2939751
    Similarity between i-th source and j-th destination vectors indicates
    a directed edge from vertex i to vertex j.

    Attributes
    ----------
    _adj_m: np.ndarray
        The adjacency matrix of the directed graph.
    _vtx_to_node: Dict
        Dictionary of key: int to value: node_id that maps vertices to original
        node_ids. _vtx_to_node[i] gives the node id of vertex i.

    Examples
    --------
    >>> from sberpm import DataHolder
    >>> from sberpm.miners import SimpleMiner
    >>> from sberpm.ml.vectorizer import GraphEmbedder
    >>> from sberpm.ml.vectorizer import katz_index, rooted_pr, \
    >>> adamic_adar, common_neighbors, vertex_similarity
    >>> vectorizer = GraphEmbedder()
    >>> embeddings = vectorizer.transform(data_holder, SimpleMiner, \
    >>> 10, katz_index, 2.5)
    >>> nodes = GraphEmbedder.get_am2nodes()
   """

    def __init__(self):
        self._vtx_to_node = None
        self._adj_m = None

    @staticmethod
    def create_adjacency(data_holder, miner: "Data Miner") \
            -> Tuple[np.ndarray, Dict]:
        """
        Uses a received miner on the data holder and computes the adjacency
        matrix of the resulting graph.

        Parameters
        ----------
        data_holder: DataHolder
            DataHolder holding the data
        miner:
            Miner used to create a graph.

        Returns
        -------
        Tuple[np.ndarray, Dict]
            Returns a computed adjacency matrix and a dictionary mapping
            vertices in the adjacency matrix to original nodes ids in the graph.

        """
        mr = miner(data_holder)
        mr.apply()
        graph = mr.graph
        vtx2node = graph.get_nodes()
        node2vtx = {v: k for k, v in enumerate(vtx2node)}
        edges_list = graph.get_edges()
        src = [x.source_node for x in edges_list]
        dst = [x.target_node for x in edges_list]
        edges = zip(src, dst)
        adj_m = np.zeros((len(vtx2node), len(vtx2node)), dtype=float)
        for src, dst in edges:
            src_i = node2vtx[src]
            dst_i = node2vtx[dst]
            adj_m[src_i][dst_i] += 1
        return adj_m, vtx2node

    def transform(self, data_holder, data_miner=SimpleMiner,
                  k: int = 0, prox_func: Callable = katz_index, *args):
        """
        Gives embeddings of nodes in the graph extracted by a received miner
        from the data holder.

        Parameters
        ----------
        data_holder: DataHolder
            A DataHolder with the log data.
        data_miner: default=SimpleMiner:
            Data miner that is going to be used.
        k: int, default=0
            Dimension to reduce the graph dimension to.
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
        Embeddings, a tuple of two np.ndarrays
            Row i of u_s (source) or u_t (target) is the embedding vector of
            vertex i.
            Similarity of vectors u_s[i] and u_t[j] for 0 <= i <= k and
            0 <= j <= k indicates a directed edge from vertex i to vertex j.
        """

        self._adj_m, self._vtx_to_node = self.create_adjacency(data_holder,
                                                               data_miner)
        assert (0 < k < self._adj_m.shape[0]), (
            "Reduced dimension should between 0 and matrix dimensions"
        )
        m_g, m_l = prox_func(self._adj_m, *args)
        assert ((m_g is not None) and (m_l is not None)), (
            "Please use proximity function that returns two correct matrix"
            "polynomials "
        )
        m_g = np.linalg.inv(m_g)
        # proximity matrix M
        M = (m_g @ m_l)
        u_s, s, u_t = svds(M, k, which="LM")
        u_s = u_s.transpose()
        n = u_t.shape[1]
        s = np.resize(np.sqrt(s), (k, n))
        u_s *= s
        u_t *= s
        embeddings = u_s, u_t
        return embeddings

    def get_am2nodes(self) -> Dict:
        """
        Returns a dictionary mapping vertices in the adjacency matrix to
        nodes.

        Returns
        -------
        Dict:
        key: int, value: node
        """
        return self._vtx_to_node
