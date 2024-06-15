from collections import Counter
from typing import List, Callable, Any

def preprocess(text: str, get_tokens: Callable[[str], Any]) -> List[str]:
    tokens = get_tokens(text)
    return [token.lemma_ for token in tokens if not token.is_stop and token.is_alpha]

# Function to find the most common words
def most_common(words: List[str], n: int) -> List[tuple[str, int]]:
    return Counter(words).most_common(n)
