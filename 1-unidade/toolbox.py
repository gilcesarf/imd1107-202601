import re
import string
import pandas as pd
from nltk.corpus import stopwords

class ZenOfPython:
    
    def __init__(self):    
        
        self.texts = ["Beautiful is better than ugly",
                      "Explicit is better than implicit",
                      "Simple is better than complex",
                      "Complex is better than complicated",
                      "Flat is better than nested",
                      "Sparse is better than dense",
                      "Readability counts",
                      "Special cases aren't special enough to break the rules",
                      "Although practicality beats purity",
                      "Errors should never pass silently",
                      "Unless explicitly silenced",
                      "In the face of ambiguity, refuse the temptation to guess",
                      "There should be one - and preferably only one - obvious way to do it",
                      "Although that way may not be obvious at first unless you're Dutch",
                      "Now is better than never",
                      "Although never is often better than right now",
                      "If the implementation is hard to explain, it's a bad idea",
                      "If the implementation is easy to explain, it may be a good idea",
                      "Namespaces are one honking great idea - let's do more of those!"]

        self.dset_tokenized = [ ]
        for text in self.texts:

            token_list = []
            for token in text.split():
                token_list.append(self.processing(token))

            self.dset_tokenized.append(token_list)

        
        self.vocab_dict = {}     
        for token_list in self.dset_tokenized:
            for token in token_list:
                if token not in self.vocab_dict:
                    token = self.processing(token) 
        
                    self.vocab_dict[token] = len(self.vocab_dict) + 1 
                
    @property
    def token_ids(self):
        
        token_ids = []            

        for token_list in self.dset_tokenized:
            vec = []                
        
            for token in token_list:
                token = self.processing(token)                   
            
                vec.append(self.vocab_dict.get(token))  
                
            token_ids.append(vec)            
        
        return token_ids

    
    @staticmethod
    def processing(token):
        '''
        Processing:
         - Converts to lowercase;
         - Removes punctuation.
        '''
        token = token.lower()
        return token.translate(str.maketrans('', '', string.punctuation))


