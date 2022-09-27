
import pandas as pd
import prodigy
from prodigy.components.loaders import TXT,JSON
from prodigy.models.matcher import PatternMatcher
import spacy
import json

import os
import sys
path = 'C:/Users/mn46/Desktop/demo_prodigy_deidentified/get_json_file/input/'
sys.path.insert(0, path) # insert path

path2 = 'C:/Users/mn46/Desktop/demo_prodigy_deidentified/get_json_file/output/'
sys.path.insert(0, path2) # insert path

df1 = pd.read_csv(os.path.join(path+ 'data_sentences.csv'), sep=',')

sentence_list = list(df1["sentences"])

print ("len(sentence_list): ", len(sentence_list))
print ("")

N = len(sentence_list)
result_list = list()
for n in range(N):
    result = dict()
    result['text'] = sentence_list[n] 
    result_list.append(result)
print ("len(result_list): ", len(result_list))
print ("")

path_ = os.path.join(path2,'data_sentences.json')

with open(path_, 'w') as outfile:
    json.dump(result_list, outfile)
    
spacy.load('en_core_web_sm')
nlp = spacy.load("en_core_web_sm")

matcher = PatternMatcher(nlp, combine_matches=True, all_examples=True).from_disk(os.path.join(path,"pattern_0317.jsonl"))

stream = JSON(path_) 

stream2 = matcher(stream)

result_list = list()
for score, eg in stream2:   
    result_list.append(eg)   
print ("len(result_list): ", len(result_list))
print ("")

path_ = os.path.join(path2,'data_sentences_matched.json')

with open(path_, 'w') as outfile:
    json.dump(result_list, outfile)
    
print ("done")

