import spacy
import os

os.system("python -m spacy download en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

def get_tokens(sentence):
    return nlp(sentence)