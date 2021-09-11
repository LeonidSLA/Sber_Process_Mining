from typing import Callable

from sberpm.miners import SimpleMiner
from ._hope_vectorizer_utils.trace_aggregation import aggregate_traces
from ._graph_embedding_utils.proximity_measures import katz_index
from ._gr_embedder import GraphEmbedder


class HopeVectorizer:
    """
    Class for event traces vectorizaition based on high order proximity
    embedding.

    Attributes
    ----------
    _source: bool
        Indicates whether the vectorization should be based on source
            or target vectors embeddings (similarity of source vector i and
            target vector j indicates a directed edge between vertex i and
            vertex j.
    """

    def __init__(self, source=True):
        """
        Parameters
        ----------
        source: bool
            Indicates whether the vectorization should be based on source
            or target vectors embeddings.
        """
        self._source = source

    def transform(self, data_holder, data_miner=SimpleMiner,
                  k: int = 0, prox_func: Callable = katz_index, *args):
        """
        Returns embeddings of the event traces in the data holder based on the
        graph mined by the miner and a high order proximity function.

        Parameters
        ----------
        data_holder:
            A data holder holding preprocessed data.
        data_miner: default=SimpleMiner
            Miner to be used to build the graph.
        k: default = 0
            Dimension to reduce the graph to.
        prox_func: Callable, default = katz_index
            Any proximity function that must return two matrix polynomials m_g
            and m_l whose modified product inv(m_g) @ m_l gives a high-order
            proximity matrix S such that S[i][j] indicates an edge between
            vertex i and vertex j.

            Five proximity functions (katz_index, rooted_pr, adamic_adar,
            common_neighbors, vertex_similarity) are implemented in
            sberpm.ml.vectorizer.
        args:
            Arguments for the proximity function.

        Returns
        -------
        embeddings: Tuple[List, List]
            Returns embeddings of the event traces in the data holder mined with
            the provided miner.
            embeddings[0] are the embeddings themselves in the format
            List[Tuple(nd.array, Tuple[str]], and embeddings[1] are the event
            traces Tuple[str] whose embeddings could not be found because of the
            reduced dimensions.
        >>> from sberpm import DataHolder
        >>> from sberpm.miners import SimpleMiner
        >>> from sberpm.ml.vectorizer import GraphEmbedder
        >>> from sberpm.ml.vectorizer import katz_index, rooted_pr, \
        >>> adamic_adar, common_neighbors, vertex_similarity
        >>> vectorizer = HopeVectorizer()
        >>> embeddings = vectorizer.transform(data_holder, SimpleMiner, \
        >>> 10, katz_index, 2.5)
        >>> trace_vectors = embeddings[0]
        >>> not_found_encodings = embeddings[1]
        """
        g_e = GraphEmbedder()
        embeddings = g_e.transform(data_holder, data_miner, k,
                                   prox_func, args)
        node2vtx = g_e.get_node_id2vtx()
        traces = data_holder.get_grouped_columns(data_holder.activity_column)
        traces2vectors = []
        vectors2traces = []
        embedding_not_found = []
        for trace in traces:
            # can only include those traces whose constituents are in the
            # embedded graph
            trace_vtx = [node2vtx[x] for x in trace if x in node2vtx]
            if not trace_vtx:
                embedding_not_found.append(trace)
            else:
                if self._source:
                    trace_vectors = [(embeddings[0])[x] for x in trace_vtx]
                else:
                    trace_vectors = [(embeddings[1])[y] for y in trace_vtx]
                trace_vector = aggregate_traces(*trace_vectors)
                traces2vectors.insert(0, trace_vector)
                vectors2traces.insert(0, trace)
        embeddings = (list(zip(traces2vectors, vectors2traces)),
                      embedding_not_found)
        return embeddings
