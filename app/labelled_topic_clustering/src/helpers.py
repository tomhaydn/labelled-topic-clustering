from collections import Counter
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from .tokenizer import get_tokens

def preprocess(text):
    tokens = get_tokens(text)
    return [token.lemma_ for token in tokens if not token.is_stop and token.is_alpha]

# Function to find the most common words
def most_common(words, n):
    return Counter(words).most_common(n)

def extract_labels(sentence_docs, num_topics=1):
    """
    Extract labels from documents in the same cluster by identifying the most
    common topics using LDA
    """
    
    # Preprocess the documents
    texts = [preprocess(doc) for doc in sentence_docs]

    print(texts)
    
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