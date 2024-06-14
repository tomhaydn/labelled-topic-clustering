import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def parse_label(topic_string, get_tokens):
    """
    Parse a label by making it more grammatically correct
    """
    nouns = []
    nsubjs = []
    verbs = []
    adjs = []

    tokens = get_tokens(topic_string)

    for token in tokens:
        # print(token, token.dep_, token.pos_, token.is_stop)
        if token.is_stop==False:
            if token.dep_ == 'nsubj':
                nsubjs.append(token.lemma_.lower())

            elif token.pos_=='NOUN' or token.pos_=='PROPN':
                nouns.append(token.lemma_.lower())
                
            elif token.pos_=='ADJ':
                adjs.append(token.lemma_.lower())

    label_words = []

    for word in nouns[:3]:
        if word not in label_words:
            label_words.append(word)

    for word in nsubjs[:1]:
        if word not in label_words:
            label_words.append(word)
    
    for word in verbs[:1]:
        if word not in label_words:
            label_words.append(word)
    
    label = '_'.join(label_words)

    return label