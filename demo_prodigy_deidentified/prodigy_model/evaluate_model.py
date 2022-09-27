# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:37:14 2022

@author: mn46
"""


import pandas as pd
import prodigy
from prodigy.components.loaders import TXT,JSON
from prodigy.models.matcher import PatternMatcher
import spacy
import json

import os
import sys

path_ = 'C:/Users/mn46/Desktop/demo_prodigy_deidentified/prodigy_api/'
sys.path.insert(0, path_) # insert path

df = pd.read_json(os.path.join(path_,'classification_output.jsonl'), lines=True)

df['PatientID'] = ''
df['Date'] = ''

import re    
df['PatientID'] = df.text.apply(lambda x: re.search('ID: (.+?) Date', x).group(1))
df['Date'] = df.text.apply(lambda x: re.search('Date: (.+?)\n', x).group(1))

df.label[df.answer == 'accept'] = 1   
df.label[df.label != 1] = 0   
        
df['keyword'] = df['label']


path = os.path.join(path_,'en_core_web_sm/model-best/')
sys.path.insert(0, path) # insert path


# from prodigy.models.textcat import TextClassifier

nlp = spacy.load(path)

#text corresponds to the Unstructured free text notes

# example here
texts = ['the patient takes keppra',
         'loading',
         'windows']
for doc in nlp.pipe(texts):
    print(doc.cats)
    
#docs = list(nlp.pipe(texts)) # check generator

# Another way = same result

docs = [nlp.tokenizer(text) for text in texts]
    
# Use textcat to get the scores for each doc
textcat = nlp.get_pipe('textcat_multilabel')
scores = textcat.predict(docs)

print(scores)