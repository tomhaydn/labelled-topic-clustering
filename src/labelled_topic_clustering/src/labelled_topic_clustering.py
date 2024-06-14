import torch
from sentence_transformers import  util

from .embeddings import get_embeddings
from .helpers import extract_labels
from .labels import parse_label
from .tokenizer import Tokenizer

DEFAULT_CLUSTER_THRESHOLD = 0.70
DEFAULT_CLUSTER_MIN_SIZE = 3

class TopicClusterer():
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
    model = ''
    hf_token = ''
    debug = False

    def __init__(self, hf_token, model, debug=False, model_cache_dir=None):
        self.hf_token = hf_token
        self.model = model
        self.debug = debug
        self.Tokenizer = Tokenizer(debug, model_cache_dir)
        
        self.Tokenizer.load_model()
        
        if debug:
            print('initialized sstc')

    def get_clusters(self, sentences=[], threshold=DEFAULT_CLUSTER_THRESHOLD, min_cluster_size=DEFAULT_CLUSTER_MIN_SIZE):
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
            A list of clusters.
        """

        if self.debug: print('building embeddings')
        embeddings = self.get_embeddings(sentences)

        embeddings_tensor = torch.FloatTensor(embeddings)
        clusters = util.community_detection(embeddings_tensor, threshold, min_cluster_size, batch_size=64)

        return clusters
    
    def get_labels_from_clusters(self, clusters, sentences):
        # Create labels for each cluster
        labelled_clusters = {}

        for cluster in clusters:
            sentence_docs = [sentences[i] for i in cluster]
            cluster_label = extract_labels(self.Tokenizer.get_tokens, sentence_docs)
            labelled_clusters[parse_label(cluster_label, self.Tokenizer.get_tokens)] = cluster
            
        return labelled_clusters
        
        
    def get_clusters_with_labels(self, sentences=[], threshold=DEFAULT_CLUSTER_THRESHOLD, min_cluster_size=DEFAULT_CLUSTER_MIN_SIZE):
        clusters = self.get_clusters(sentences, threshold, min_cluster_size)
        return self.get_labels_from_clusters(clusters, sentences)

    def get_embeddings(self, sentences = []):
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
        embeddings = get_embeddings(sentences, self.hf_token, self.model)
        return embeddings