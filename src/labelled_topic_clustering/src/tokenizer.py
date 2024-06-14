import spacy
import os

MODEL_NAME = "en_core_web_sm"

class Tokenizer():
    model_initialise = False
    nlp = None
    model_cache_dir = None
    
    def __init__(self, debug=False, model_cache_dir=None):
        self.model_cache_dir = model_cache_dir
        if debug:
            print('initialized tokenizer')
            
    def get_tokens(self, sentence):
        return self.nlp(sentence)
    
    def load_model(self):
        model_cache_full_path = os.path.join(self.model_cache_dir, MODEL_NAME)
        
        if os.path.exists(model_cache_full_path):
            self.nlp = spacy.load(model_cache_full_path)
        else:
            os.system(f"python -m spacy download {MODEL_NAME}")
            self.nlp = spacy.load(MODEL_NAME)
            self.nlp.to_disk(model_cache_full_path)
        
        