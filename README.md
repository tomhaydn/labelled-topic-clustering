# Super Simple Topic Clustering

Super Simple Topic Clustering is as the name suggests, feed it an array of sentences and it will cluster them with human-readable names.

## Quick Start

`pip install labelled-topic-clustering`

```
sentences = ["the weather is good", "my dog ate my homework"] # json.load(f)
HF_TOKEN = '...'
MODEL_ID="sentence-transformers/all-mpnet-base-v2"

clusterer = TopicClusterer(HF_TOKEN, MODEL_ID, debug=True, model_cache_dir="an/absolute/path/that/is/a/good/place/to/store/large/files")

clusters = clusterer.get_clusters(sentences, 0.5, 2)
print(clusters)

clusters_labelled = clusterer.get_labels_from_clusters(clusters, sentences)
print(clusters_labelled)
```
