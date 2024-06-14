import spacy
import os

class Tokenizer():
    model_initialise = False
    nlp = None
    
    def __init__(self, debug=False):
        if debug:
            print('initialized tokenizer')
            
    def get_tokens(self, sentence):
        return self.nlp(sentence)
    
    def load_model(self):
        os.system("python -m spacy download en_core_web_sm")
        self.nlp = spacy.load("en_core_web_sm")
        