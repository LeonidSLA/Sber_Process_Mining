import numpy as np
from scipy.spatial.distance import cdist
from ...ml.text import TextPreprocessing
from ...ml.vectorizer import TextWord2Vec
from ...ml.processes import GraphClustering


class KMeansSummarizer:
    """
    Simple summarizer based on k-means algo.

    Parameters
    ----------
    texts: pd.Series or list
        List of texts for summarization.

    clear_texts: list, default=None
        List of processed texts using the built-in library tool or similar tool.

    w2v_model: sberpm.ml.vectorizer.TextWord2Vec, default=None
        Pretrained word2vec model.

    centroids: np.ndarray, default=None
        Array of cluster centers obtained after clustering by the k-means algo.

    labels: np.ndarray, default=None
        Clustering label list.


    """

    def __init__(self, texts, max_cluster_num=20, clear_texts=None, w2v_model=None, centroids=None, labels=None):
        self.w2v = None
        self.clear_text = clear_texts if clear_texts else None
        self.texts = texts

        if type(self.texts) == 'list':
            self.texts = pd.Series(self.texts)

        self.texts = self.texts.reset_index(drop=True)
        self.centroids = centroids if centroids else None
        self.labels = labels if labels else None

        if w2v_model is None or self.clear_text is None:
            self._fit_w2v()
        else:
            self.w2v = w2v_model

        self.embeddings = self.w2v.transform(self.clear_text)

        if centroids is None or labels is None:
            text_clusterer = GraphClustering()
            text_clusterer.fit(self.embeddings, max_cluster_num=max_cluster_num)
            self.labels = text_clusterer.predict(self.embeddings)
            self.centroids = text_clusterer._model.cluster_centers_

    def _fit_w2v(self):

        preprocessing = TextPreprocessing()
        self.clear_text = preprocessing.transform(self.texts)
        self.w2v = TextWord2Vec(size=100)
        self.w2v = self.w2v.fit(self.clear_text)

    def summarize(self, n_sentences=5, algo='advanced'):
        """
        Summarizes texts by clusters. If the parameter  'algorithm' == 'simple', then the method will return
        n = 'n_sentences' of texts evenly distant from the centers of the clusters. If 'algorithm' == 'advanced',
        then each initial cluster will be additionally clustered, and as an abstract for of each cluster,
        the texts that are the least distant from the centers of the subclusters will be taken .

        Parameters
        ----------
        n_sentences: int, default=5
            Number of texts in the abstract.
            Note: parameter is specified only if 'algorithm'=='simple'.

        algo: {'simple','advanced'}, default='simple'
            Algorithm used for summarization.

        Returns
        -------
        text_dict: dict
            Dictionary, where the keys are the labels of the clusters,
            the values are a short summary of the cluster.
        """
        text_dict = {}
        if algo == 'simple':

            for i in range(len(self.centroids)):
                embeddings_set = self.embeddings[self.labels == i]
                target_idx = np.array(np.where(self.labels == i))[0, :]
                center_set = np.reshape(self.centroids[i], (1, len(self.centroids[i])))
                dist_matrix = cdist(embeddings_set, center_set, metric='cosine')
                ind_set = (
                    np.argsort(dist_matrix, axis=0)[np.linspace(0, len(embeddings_set) - 1, num=n_sentences, dtype=int)])
                final_idx = target_idx[ind_set[:, 0]]
                text_dict[i] = ' || '.join(self.texts[final_idx])

        elif algo == 'advanced':

            for i in range(len(self.centroids)):
                embeddings_set = self.embeddings[self.labels == i]
                target_idx = np.array(np.where(self.labels == i))[0, :]
                text_clusterer = GraphClustering()
                text_clusterer.fit(embeddings_set, max_cluster_num=15)
                centers_subset = text_clusterer._model.cluster_centers_
                dist_matrix = cdist(embeddings_set, centers_subset, metric='cosine')
                ind_set = np.argsort(dist_matrix, axis=0)[0]
                final_idx = target_idx[ind_set]
                text_dict[i] = ' || '.join(self.texts[final_idx])
        else:
            raise ValueError(f"Algo should be 'simple' or 'advanced', but got '{algo}' instead.")

        return text_dict

    def get_keywords(self, n=10):
        """
        Returns a list of keywords describing each cluster sorted in descending order of importance.

        Parameters
        ----------
        n: int, default=10
            Number of keywords returned.

        Returns
        -------
        token_dict: dict
            Dictionary, where the keys are the labels of the clusters, the values are the keywords.
        """
        token_dict = {}

        for i in range(len(self.centroids)):
            token_dict[i] = self.w2v.most_similar(target=[self.centroids[i]], n=n)

        return token_dict
