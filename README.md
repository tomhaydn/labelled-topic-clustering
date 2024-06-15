# Labelled Topic Clustering

Labelled Topic Clustering is as the name suggests, feed it an array of sentences and it will cluster them with human-readable names.

The aim of this project is to make it as **easy-as-possible** to:

1. generate topic clusters on a text dataset using a cosine-similarity approach.
2. get human-readable labels for those clusters

![labelled topic clustering approach](https://github.com/tomhaydn/labelled-topic-clustering/blob/main/docs/diagram-1.png)

## Installation

To use the TopicClusterer class, you need to install the required packages. Assuming you have a package manager like pip, you can install the dependencies as follows:

`pip install torch sentence-transformers spacy`

## Usage

1. Initialize the TopicClusterer:

```
from topic_clusterer import TopicClusterer

hf_token = "your_hugging_face_token"
model = "your_model_name"

clusterer = TopicClusterer(hf_token, model, debug=True)
```

2. Get clusters:

```
sentences = [
    "the weather is great",
    "This is some perfect weather",
    "we're having some really good weather",
    "my dog ate my homework",
    "why do dogs love homework?",
    "dog keeps devouring my homework"
]

clusters = clusterer.get_clusters(sentences)
```

#### Example Output

```
[[0, 1, 2], [3, 4, 5]]
```

`clusters` will be a 2d array representing clusters with sentence indicies for the original dataset

3. Get labels from clusters:

```
clusters_labelled = clusterer.get_labels_from_clusters(clusters, sentences)
```

#### Example Output

```
{'Weather great perfect': [0, 1, 2], 'Dog eat homework': [3, 4, 5]}
```

`clusters_labelled` is a dictionary where the keys are topic labels, and the values are arrays of sentence indices corresponding to the original dataset.

> You can also just get it all at once:

```
# Get clusters with labels
labelled_clusters = clusterer.get_clusters_with_labels(sentences)
print(labelled_clusters)
```

# Looing Forward

I have done virtually no performance testing as I wrote this once and it was all I needed for a side project.

Some ideas to work on:

- Allow custom tokenizers
- Benchmark performance on large datasets
- Allow for feature extraction locally

```

```
