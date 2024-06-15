import os

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from .helpers import most_common, preprocess

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

def parse_label(topic_string, get_tokens, topic_length=3):
    """
    Parse a label by making it more grammatically correct
    """
    tokens = get_tokens(topic_string)

    nouns = []
    verbs = []
    adjs = []
    others = []

    for token in tokens:
        if not token.is_stop and token.is_alpha:
            if token.pos_ in ['NOUN', 'PROPN']:
                nouns.append(token)
            elif token.pos_ == 'VERB':
                verbs.append(token)
            elif token.pos_ == 'ADJ':
                adjs.append(token)
            else:
                others.append(token)

    label_tokens = []

    if nouns:
        label_tokens.append(nouns.pop(0))  # Subject
    if verbs:
        label_tokens.append(verbs.pop(0))  # Verb
    if nouns:
        label_tokens.append(nouns.pop(0))  # Object

    while len(label_tokens) < topic_length and (nouns or verbs or adjs or others):
        if nouns:
            label_tokens.append(nouns.pop(0))
        elif verbs:
            label_tokens.append(verbs.pop(0))
        elif adjs:
            label_tokens.append(adjs.pop(0))
        else:
            label_tokens.append(others.pop(0))

    label = ' '.join([token.text for token in label_tokens[:topic_length]])

    if label:
        label = label[0].upper() + label[1:]

    return label


def extract_labels(get_tokens, sentence_docs, num_topics=1):
    """
    Extract labels from documents in the same cluster by identifying the most
    common topics using LDA
    """
    
    # Preprocess the documents
    texts = [preprocess(doc, get_tokens) for doc in sentence_docs]
    
    # Create a dictionary representation of the documents
    dictionary = Dictionary(texts)
    
    # Create a bag of words representation of the corpus using the dictionary
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train an LDA model
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42)

    # Get the most common topic
    topics = lda.show_topics(formatted=False)
    topic_tuple = most_common([word for word, _ in topics[0][1]], 10)
    topic_string = ' '.join([str(word[0]) for word in topic_tuple])
    
    return topic_string