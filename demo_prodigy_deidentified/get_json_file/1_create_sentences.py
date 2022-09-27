# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 19:55:39 2022

@author: mn46
"""

import sys
import os
import numpy as np  
import pandas as pd
from collections import OrderedDict

path = 'C:/Users/mn46/Desktop/demo_prodigy_deidentified/get_json_file/input/'
sys.path.insert(0, path) # insert path


df = pd.read_csv(os.path.join(path,'Ground_truth_before_filters.csv'), sep=',')

df['Unstructured'] = df['Unstructured'].str.lower()
df.Unstructured = df.Unstructured.apply(lambda x: " ".join(x.split()))

df = df.reset_index(drop=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer

antiEpilepsyBagOfWords = {'evid': {'not', 'evid', 'diagnosi', 'epilepsi'},
                      'recommend': {'not', 'recommend', 'antiepilept', 'medic'},
                      'defer sz': {'defer', 'anti', 'seizur'},
                      'defer med': {'defer', 'anti', 'epilept'},
                      'refer': {'referr', 'gener', 'neurolog'},
                      'follow up': {'not', 'requir', 'follow', 'up'},
                      'followup': {'not', 'requir', 'followup'},
                      'cannot': {'cannot', 'event', 'epilept'},
                      'pnes': {'pnes'},
                      'nosz': {'no', 'seizur', 'event'},
                      'unlikely': {'unlik', 'seizur'},
                      'fnd': {'function', 'neurolog', 'disord'},
                      'migraine': {'migrain'},
                      'anxiety': {'anxieti'},
                      'syncope': {'syncop'},
                      'cd': {'convers', 'disord'},
                      'psycho': {'psychogen'},
                      'risk': {'not', 'have', 'seizur', 'risk', 'factor'},
                      'sleep': {'sleep', 'disord'},
                      'apnea': {'sleep', 'apnea'},
                      'test': {'not', 'recommend', 'test'},
                      'suspicion': {'low', 'suspicion', 'seizur'},
                      'tremor': {'physiolog', 'tremor'},
                      '"seizures"': {"''", 'seizur'},
                      'fn': {'function', 'neurolog'},
                      'vasovagal': {'vasovag'},
                      'pcp': {'defer', 'primary', 'care', 'physician'},
                      'definition': {'not', 'meet', 'definit', 'epilepsi'},
                      'support': {'not', 'support', 'diagnosi', 'epilepsi'},
                      'amnesia': {'amnesia'},
                      'provoke': {'provok', 'seizur'},
                      'depression': {'dispress'},
                      'shiver': {'shiver'},
                      'arrest': {'cardiac', 'arrest'},
                      'noanti': {'no', 'anti', 'seizur', 'medic'},
                      'neuropathy': {'neuropathi'},
                      'neuropathic': {'neuropath'},
                      'meningioma': {'meningioma'},
                      'holdoff': {'hold', 'off', 'start', 'anti', 'epilept'},
                      'diabetes': {'diabet'},
                      'neurosarcoidosis': {'neurosarcoidosi'},
                      'sdh': {'sdh'},
                      'postoper': {'post', 'oper'},
                      'hemorrhage': {'traumat', 'hemorrhag'},
                      'concern': {'low', 'concern', 'seizur'},
                      'convince': {'not', 'convinc', 'seizur'},
                      'follow': {'not', 'need', 'follow', 'epilepsi'},
                      'notfollowup': {'not', 'need', 'followup'},
                      'start': {'not', 'start', 'antiepilept', 'medic'},
                      'startsz': {'not', 'start', 'antiseizur', 'medic'},
                      'cause': {'unlik', 'epilepsi'},
                      'trauma': {'trauma'},
                      'traumatic': {'traumat'},
                      'hematoma': {'hematoma'},
                      'abscess': {'brain', 'abscess'},
                      'hold': {'hold', 'off', 'medic'},
                      'postop': {'postop'},
                      'single': {'singl', 'seizur'},
                      'funcevents': {'function', 'event'},
                      'asneeded': {'follow', 'up', 'as', 'need'},
                      'asneededfollow': {'followup', 'as', 'need'},
                      'referpsy': {'referr', 'psychiatri'},
                      'defermed': {'defer', 'medic'},
                      'noconcern': {'no', 'concern', 'seizur'},
                      'acute': {'acut', 'symptomat', 'seizur'},
                      'first': {'first', 'time', 'seizur'},
                      'lifetime': {'one', 'lifetim', 'seizur'},
                      'evidence': {'no', 'evid', 'seizur'},
                      'sudep': {'sudep'},
                      'meet': {'not', 'meet', 'epilepsi'},
                      'notneedmedic': {'not', 'need', 'medic'},
                      'jacobsen': {'jacobsen', 'syndrom'},
                      'alcohol': {'excess', 'alcohol'},
                      'exam': {'normal', 'neurolog', 'exam'},
                      'mri': {'normal', 'mri'},
                      'eeg':{'normal', 'eeg'},
                      'eprisk': {'no', 'epilepsi', 'risk'},
                      'factors': {'no', 'epilepsi', 'risk', 'factor'},
                      'epileptiform': {'no', 'epileptiform', 'abnorm'},
                      'psychiatric': {'psychiatr'},
                      'fentanyl': {'fentanyl'},
                      'bipolar': {'bipolar'},
                      'not have': {'not', 'have', 'epilepsi'},
                      'bite': {'no', 'bite'},
                      'incontinence': {'no', 'incontin'},
                      'lowthres': {'low', 'seizur', 'threshold'},
                      'lowerthres': {'lower', 'seizur', 'threshold'},
                      'antisz': {'no', 'antiseizur', 'medic'},
                      'had': {'not', 'had', 'seizur'},
                      'nonepileptic': {'nonepilept'},
                      'chemo': {'chemo'},
                      'chemotherapy': {'chemotherapi'},
                      'epileptogenic': {'no', 'epileptogen', 'abnorm'},
                      'numb': {'numb'},
                      'surgery': {'surgeri'},
                      'discharge': {'discharg'},
                      'nonepileptiform': {'nonepileptiform'},
                      'non epileptiform': {'non', 'epileptiform'},
                      'not epileptic': {'not', 'epilept'},
                      'dementia': {'dementia'},
                      'think': {'not', 'think', 'epilepsi'},
                      'diagnose': {'no', 'diagnosi', 'epilepsi'}}

proEvidences = {'both': {'both', 'epilepsi', 'pnes'},
              'mixed dis': {'mix', 'disord'},
              'ictal': {'ictal'},
            'aura': {'aura'},
            'convulse': {'convuls'},
            'breakthrough': {'breakthrough', 'seizur'},
            'focal': {'focal'},
            'idiopathic': {'idiopath', 'general', 'epilepsi'},
            'history': {'histori', 'seizur'},
            'hx': {'hx', 'seizur'},
            'complex': {'complex', 'seizur'},
            'partial': {'partial', 'seizur'},
            'myoclonic': {'myoclon'},
            'generalized': {'general', 'seizur'},
            'continue': {'continu', 'on'},
            'drive': {'drive', 'month'},
            'szdrive': {'drive', 'seizur'},
            'deja': {'deja', 'vu'},
            'seizurefree': {'seizurefre'},
            'szfree': {'szfree'},
            'seizure free': {'seizur', 'free'},
            'sz free':{'sz', 'free'},
            'frontallobe': {'frontal', 'lobe'},
            'nocturnal': {'nocturn'},
            'febrile': {'febril'},
            'perinatal': {'perinat', 'complic'},
            'control': {'control'}, 
            'monotherapy': {'monotherapi'},
            'absence': {'absenc', 'seizur'},
            'dejavu': {'dejavu'},
            'postictal': {'postict', 'confus'},
            'tonicclonic': {'tonniclon'},
            'tonic clonic': {'tonic', 'clonic'}}

aeds = ['acetazolamid', 'acth',
    'acthar', 'brivaracetam',
    'briviact', 'cannabidiol' , 'epidiolex',
    'carbamazepin', 'cbz', 'epitol', 'tegretol', 'equetro', 'teril',
     'carbatrol', 'tegretol', 'epitol', 'cenobam', 'xcopri',
     'clobazam', 'frisium', 'onfi', 'sympazan', 'clonazepam',
     'epitril', 'klonopin', 'rivotril', 'clorazep', 'tranxen',
     'xene', 'diazepam', 'valium' , 'diamox',
     'diastat', 'divalproex', 'depakot', 'eslicarbazepin', 'aptiom',
     'ethosuximid', 'zarontin', 'ethotoin', 'ezogabin', 'potiga',
     'felbam', 'felbatol', 'gabapentin', 'neurontin', 'gralis',
     'horiz', 'lacosamid', 'vimpat', 'lamotrigin', 'lamict',
     'levetiracetam', 'ltg', 'ige', 'tpm', 'oxc', 'lev', 'keppra', 'roweepra', 'spritam',
     'elepsia', 'lorazepam', 'ativan', 'methsuximid', 'methosuximid',
     'celontin', 'oxcarbazepin', 'trilept', 'oxtellar xr', 'perampanel',
    'fycompa', 'phenobarbit', 'luminol', 'lumin', 'phenytoin',
     'epanutin', 'dilantin', 'phenytek', 'pregabalin', 'lyrica',
     'primidon', 'mysolin', 'rufinamid', 'banzel', 'inovelon', 'percocet',
     'stiripentol', 'diacomit', 'tiagabin', 'gabitril', 'topiram', 'topamax',
     'topiram',  'qudexi', 'trokendi', 'valproat', 'valproic', 'wellbutrin',
     'convulex', 'depacon', 'depaken', 'orfiril', 'valpor', 'valprosid',
     'depakot', 'vigabatrin', 'sabril', 'vigadron', 'zonisamid', 'zonegran', 'xanax', 'no aeds']

stemmer = SnowballStemmer(language='english')
sentenceList = list()

for index in range(len(df)):
    print(index)
    note = df.iloc[index].get('Unstructured')
    note = str(note)
    sentences = sent_tokenize(note)
    foundSentences = list()
    checkSentences = set()
    foundSentences.append('Patient ID: ' + df.iloc[index].get('PatientID') + ' Date: ' + df.iloc[index].get('Date') + '\n\n')
    for sentence in sentences:
        words = word_tokenize(sentence)
        stem_words = []

        for w in words:
            x = stemmer.stem(w)
            stem_words.append(x)

        for word in stem_words:
            if word in aeds:
                if sentence not in checkSentences:
                    checkSentences.add(sentence)
                    addSentence = 's' + str(index) + '_sentence_' + str(len(foundSentences)) + ' ' + sentence + '\n\n'
                    foundSentences.append(addSentence)

        for bag in antiEpilepsyBagOfWords:
            if antiEpilepsyBagOfWords[bag].issubset(stem_words):
                if sentence not in checkSentences:
                    checkSentences.add(sentence)
                    addSentence = 's' + str(index) + '_sentence_' + str(len(foundSentences)) + ' ' + sentence + '\n\n'
                    foundSentences.append(addSentence)
                
        for bag in proEvidences:
            if proEvidences[bag].issubset(stem_words):
                if sentence not in checkSentences:
                    checkSentences.add(sentence)
                    addSentence = 's' + str(index) + '_sentence_' + str(len(foundSentences)) + ' ' + sentence + '\n\n'
                    foundSentences.append(addSentence)
    
    single = '\n'.join(foundSentences)
    sentenceList.append(single)
    
df['sentences'] = sentenceList
df = df.drop(columns=['MRN', 'patient_has_epilepsy', 'Unstructured']) 

df.to_csv(os.path.join(path,'data_sentences.csv'), sep=',')

