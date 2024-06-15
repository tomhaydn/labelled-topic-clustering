import torch
from sentence_transformers import  util

from .embeddings import get_embeddings
from .labels import parse_label, extract_labels
from .tokenizer import Tokenizer

DEFAULT_CLUSTER_THRESHOLD = 0.70
DEFAULT_CLUSTER_MIN_SIZE = 3

class TopicClusterer:
    """
    TopicClusterer is a class used to cluster topics based on sentence embeddings.

    Attributes:
    -----------
    model : str
        The name of the model to be used for embeddings.
    hf_token : str
        The Hugging Face token for authentication.
    debug : bool
        Flag to enable or disable debug mode.
    """
    model: str = ''
    hf_token: str = ''
    debug: bool = False

    def __init__(self, hf_token: str, model: str, debug: bool = False, model_cache_dir: str = None):
        """
        Initializes the TopicClusterer with the provided Hugging Face token, 
        model name, and debug flag.

        Parameters:
        -----------
        hf_token : str
            The Hugging Face token for authentication.
        model : str
            The name of the model to be used for embeddings.
        debug : bool, optional
            Flag to enable or disable debug mode (default is False).
        model_cache_dir : str, optional
            The directory to cache the model (default is None).
        """
        self.hf_token = hf_token
        self.model = model
        self.debug = debug
        self.tokenizer = Tokenizer(debug, model_cache_dir)

        self.tokenizer.load_model()

        if debug:
            print('initialized sstc')

    def get_clusters(
        self,
        sentences,
        threshold: float = DEFAULT_CLUSTER_THRESHOLD,
        min_cluster_size: int = DEFAULT_CLUSTER_MIN_SIZE
    ) -> list[list[int]]:
        """
        Get topic clusters from a list of sentences.

        Parameters:
        -----------
        sentences : list of str, optional
            The list of sentences to be clustered (default is an empty list).
        threshold : float, optional
            The threshold for clustering (default is DEFAULT_CLUSTER_THRESHOLD).
        min_cluster_size : int, optional
            The minimum size of a cluster (default is DEFAULT_CLUSTER_MIN_SIZE).

        Returns:
        --------
        list
            A list of clusters, where each cluster is a list of sentence indices.
        """

        if self.debug:
            print('building embeddings')
        embeddings = self.get_embeddings(sentences)

        embeddings_tensor = torch.FloatTensor(embeddings)
        clusters = util.community_detection(
            embeddings_tensor,
            threshold,
            min_cluster_size,
            batch_size=64
        )

        return clusters

    def get_labels_from_clusters(
        self,
        clusters: list[list[int]],
        sentences: list[str]
    ) -> dict:
        """
        Create labels for each cluster of sentences.

        Parameters:
        -----------
        clusters : list of list of int
            The clusters of sentence indices.
        sentences : list of str
            The list of sentences.

        Returns:
        --------
        dict
            A dictionary where keys are cluster labels and values are lists of sentence indices.
        """
        labelled_clusters = {}

        for cluster in clusters:
            sentence_docs = [sentences[i] for i in cluster]
            cluster_label = extract_labels(self.tokenizer.get_tokens, sentence_docs)
            labelled_clusters[parse_label(cluster_label, self.tokenizer.get_tokens)] = cluster

        return labelled_clusters


    def get_clusters_with_labels(
        self,
        sentences,
        threshold: float = DEFAULT_CLUSTER_THRESHOLD,
        min_cluster_size: int = DEFAULT_CLUSTER_MIN_SIZE
    ) -> dict:
        """
        Get topic clusters with labels from a list of sentences.

        Parameters:
        -----------
        sentences : list of str, optional
            The list of sentences to be clustered (default is an empty list).
        threshold : float, optional
            The threshold for clustering (default is DEFAULT_CLUSTER_THRESHOLD).
        min_cluster_size : int, optional
            The minimum size of a cluster (default is DEFAULT_CLUSTER_MIN_SIZE).

        Returns:
        --------
        dict
            A dictionary where keys are cluster labels and values are lists of sentence indices.
        """
        clusters = self.get_clusters(sentences, threshold, min_cluster_size)
        return self.get_labels_from_clusters(clusters, sentences)

    def get_embeddings(self, sentences) -> list:
        """
        Get embeddings for a list of sentences.

        Parameters:
        -----------
        sentences : list of str, optional
            The list of sentences to get embeddings for (default is an empty list).

        Returns:
        --------
        list
            A list of embeddings.
        """
        embeddings = get_embeddings(sentences, self.hf_token, self.model, self.debug)
        return embeddings
    