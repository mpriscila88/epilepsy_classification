# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:29:05 2022

@author: cdac
"""

import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import FunctionTransformer

from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (make_scorer, roc_auc_score, roc_curve, precision_recall_curve) 
        
import dill
import sys
import os


#%% Import data ###############################################################

path = 'C:/Users/mn46/Desktop/eprojects/Pycode/code_modeling_deidentified/'
sys.path.insert(0, path) # insert path

df = pd.read_csv(os.path.join(path,'dataset.csv'), index_col=0)
 


#%% Features for modeling #########################################

outcome = 'outcome'
labels = ['NO', 'YES']


df['Sex'][df['Sex'].astype(str) == 'Male'] = 1
df['Sex'][df['Sex'].astype(str) != '1'] = 0
df['Sex'] = df['Sex'].astype(int)


#Split patients in train/test #################################################

e = df[['Patient_deidentified']].drop_duplicates()

e_train, e_test = train_test_split(e, test_size=0.3, random_state=42) 

# Assign all respective MRNs encounters in train and test
df_train =  df[df.Patient_deidentified.isin(e_train.Patient_deidentified)] 
df_test =  df[df.Patient_deidentified.isin(e_test.Patient_deidentified)] 

X_train = df_train.drop(columns=['Patient_deidentified','patient_has_epilepsy','Date_deidentified'])
X_test = df_test.drop(columns=['Patient_deidentified','patient_has_epilepsy','Date_deidentified'])

y_train = df_train['patient_has_epilepsy'].astype(int)
y_test = df_test['patient_has_epilepsy'].astype(int) 

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

cols = ['n_icds', 'n_meds', 'Age']

for col in cols:
    X_test[col] = np.round(scale_range (X_test[col], np.min(X_train[col]), np.max(X_train[col])))
    X_test[col] = (X_test[col] - np.min(X_train[col])) / (np.max(X_train[col]) - np.min(X_train[col]))
    X_train[col] = (X_train[col] - np.min(X_train[col])) / (np.max(X_train[col]) - np.min(X_train[col]))
    
#Pipelines ###################################################################

get_numeric_data = FunctionTransformer(lambda x: x[X_train.columns], validate=False)

clf = XGBClassifier(random_state = 42, max_iter=1000)

num_pipe = Pipeline([
  ('select_num', get_numeric_data),
  ])

full_pipeline = Pipeline([
    ('feat_union', FeatureUnion(transformer_list=[
            ('num_pipeline', num_pipe),
          ])),
    ('clf', clf)
    ])


lst_params = {'clf__n_estimators': [100,150,200,250,300,350],
              'clf__learning_rate': [0.01,0.05, 0.06,0.07,0.08,0.09,0.1],
              'clf__colsample_bytree': [0.3,0.4,0.5,0.6,0.7,0.8],
              'clf__subsample': [0.8,0.9,1],
              'clf__max_depth': [2,3,4,5],
              'clf__gamma': [0,1,5],
              'clf__warm_start':[True,False]}

random_search = RandomizedSearchCV(full_pipeline, param_distributions=lst_params, n_iter=100, cv=5, refit = True, n_jobs=-1, verbose=1, random_state = 42)

#------------------------------------------------------------------------
# Train and test
#------------------------------------------------------------------------

import sys
sys.setrecursionlimit(10000)

random_search.fit(X_train, y_train) 

clf = random_search.best_estimator_
    
#------------------------------------------------------------------------
# Performance
#------------------------------------------------------------------------

y_train_pred = clf.predict_proba(X_train*1)[:,1]
y_test_pred = clf.predict_proba(X_test*1)[:,1]

#%% Threshold functions #######################################################
 
# def optimal_threshold_auc(target, predicted):
#     fpr, tpr, threshold = roc_curve(target, predicted)
#     i = np.arange(len(tpr))
#     roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
#     roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
#     return list(roc_t['threshold'])

# Optimal Threshold for Precision-Recall Curve (Imbalanced Classification)
    
def optimal_threshold_auc(target, predicted):
    precision, recall, threshold = precision_recall_curve(target, predicted)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    return threshold[ix]
  
# Threshold in train
threshold = optimal_threshold_auc(y_train, y_train_pred)

y_pred = (clf.predict_proba(X_test*1)[:,1] >= threshold).astype(np.int)

probs = clf.predict_proba(X_test)


from performance_binary2 import perf

boot_all_micro, boot_all_macro, boot_label = perf(y_test, y_pred, probs, labels)

#%%##########################################################################
