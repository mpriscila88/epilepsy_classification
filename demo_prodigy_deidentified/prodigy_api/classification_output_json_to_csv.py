# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:12:02 2022

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
path = 'C:/Users/mn46/Desktop/demo_prodigy_deidentified/prodigy_api/'
sys.path.insert(0, path) # insert path

df = pd.read_json(os.path.join(path,'classification_output.jsonl'), lines=True)

df['PatientID'] = ''
df['Date'] = ''

import re    
df['PatientID'] = df.text.apply(lambda x: re.search('ID: (.+?) Date', x).group(1))
df['Date'] = df.text.apply(lambda x: re.search('Date: (.+?)\n', x).group(1))


df.to_csv(os.path.join(path,'classification_output.csv'), sep=',')

