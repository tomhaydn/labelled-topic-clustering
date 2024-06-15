from collections import Counter

def preprocess(text, get_tokens):
    tokens = get_tokens(text)
    return [token.lemma_ for token in tokens if not token.is_stop and token.is_alpha]

# Function to find the most common words
def most_common(words, n):
    return Counter(words).most_common(n)