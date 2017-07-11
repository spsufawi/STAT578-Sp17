# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:48:07 2017

@author: Xinchen
"""
import os
import pandas as pd
import numpy as np
import string
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words
from copy import deepcopy
import xgboost as xgb

###############Load Data###############
os.chdir("C:/Users/Xinchen/Desktop/stat578 Qu")
dt = pd.read_csv("quora_duplicate_questions.tsv", sep='\t', index_col="id",\
                 usecols = [0,3,4,5])

###############Preprocessing###############

dt= dt.dropna()
dt = dt.reset_index(drop = True)

###############Duplicate Questions Detection###############

#Processing
Stopwords = [set(stopwords.words('english') + get_stop_words('english'))]  
punctuation = string.punctuation+'``' + "''" + "..." 

def tokenize(text):
     text = text.lower()
     tokens = [token for token in text.split()]
     tokens = [token for token in text.split() if token not in Stopwords[0]]
     tokens = word_tokenize(" ".join(tokens))
     tokens = [token for token in tokens if token not in punctuation]
     tokens = [token for token in tokens if token not in ["'s","n't","ca","doe"]]
     stemmer = EnglishStemmer()
     stems = map(stemmer.stem, tokens)
     lemmer = WordNetLemmatizer()
     lemms = map(lemmer.lemmatize, stems)
     return  list(lemms)

cl_dt1 = []
cl_dt2 = []
index1 = []
index2 = []

for i in range(len(dt)):
    cl_dt1.append(tokenize(dt['question1'][i]))     
    if len(cl_dt1[i]) == 0:
          index1.append(i)
    cl_dt2.append(tokenize(dt['question2'][i]))
    if len(cl_dt1[i]) == 0:
          index2.append(i)

c_dt1 = deepcopy(cl_dt1)  #in case
c_dt2 = deepcopy(cl_dt2)

index1 = [q for q in range(len(cl_dt1)) if len(cl_dt1[q]) == 0]
index2 = [q for q in range(len(cl_dt2)) if len(cl_dt2[q]) == 0]

###############word similarity###############
wordVmodel = gensim.models.Word2Vec(cl_dt1+cl_dt2, window = 5, min_count = 1)

similarity = np.zeros(len(cl_dt1))
for k in range(len(cl_dt1)):
     for i in cl_dt1[k]:
          for j in cl_dt2[k]:
               similarity[k] += wordVmodel.similarity(i,j)
     if len(cl_dt1[k])!=0 and len(cl_dt2[k])!=0:
           similarity[k] = similarity[k]/(len(cl_dt1[k])+len(cl_dt1[k]))

###############word_share###############
q1len = np.zeros(len(cl_dt1))
q2len = np.zeros(len(cl_dt2))
wordshare = np.zeros(len(cl_dt1))
for i in range(len(cl_dt1)):
     if len(cl_dt1[i])!= 0 and len(cl_dt2[i])!=0 :
          q1len[i] = len(cl_dt1[i])
          q2len[i] = len(cl_dt2[i])
          wordshare[i] = len(set(cl_dt1[i] + \
                   cl_dt2[i]))/(q1len[i] + q2len[i])
          
###############weight_share###############
tf1 = TfidfVectorizer(lowercase = True, tokenizer = tokenize)
tf2 = TfidfVectorizer(lowercase = True, tokenizer = tokenize)
tfidf1 = tf1.fit_transform(dt['question1'])
tfidf2 = tf2.fit_transform(dt['question2'])

weight_share = np.zeros(len(dt))
for k in range(len(dt)):
     weight_share[k] = tfidf1[k,].sum() - tfidf2[k,].sum()
###############shared_word###############
shared_word = [[] for x in range(len(dt))]

for k in range(len(dt)):
     for i in cl_dt1[k]:
          for j in cl_dt2[k]:
               if i == j:
                    shared_word[k].append(i)
     shared_word[k] = list(set(shared_word[k]))

index_s1 =[[] for x in range(len(dt))] 
index_s2 = [[] for x in range(len(dt))] 

###############Add the new features to the data###############
dt['q1len'] = q1len
dt['q2len'] = q2len
dt['wordshare'] = wordshare
dt['similarity'] = similarity
dt['diff1'] = dt['question1'].str.len() 
dt['diff2'] = dt['question2'].str.len()
dt['weight_share'] = weight_share

###############split the dataset###############
X_train, X_test, Y_train, Y_test = train_test_split(\
          dt[['diff1','diff2','q1len','q2len','wordshare','similarity','weight_share']],\
                                dt['is_duplicate'], test_size = 0.25)
def gv(predictions, Y, prob):
     tr = np.zeros(len(Y))
     accy = 0
     for i in range(len(Y)):
          if predictions[i] == Y.iloc[i]:
              tr[i] = 1
          else:
               tr[i] = 0
     accy = np.mean(predictions==Y.as_matrix().ravel())
     logloss = log_loss(tr,prob)
     return accy, logloss
                 
###############Logistic regression###############
model = LogisticRegression()
model = model.fit(X_train, Y_train.as_matrix())
prob = model.predict_proba(X_test.as_matrix())
prediction = model.predict(X_test.as_matrix())

gv(prediction, Y_test, prob)

###############Random forest###############

rf = RandomForestClassifier()        
rf.fit(X_train, Y_train.as_matrix())
prob_rf = rf.predict_proba(X_test.as_matrix())
prediction_rf = rf.predict(X_test.as_matrix())


gv(prediction_rf, Y_test, prob_rf)

###############GBM###############
gbm = GradientBoostingClassifier()
gbm.fit(X_train, Y_train.as_matrix())
prob_gbm =gbm.predict_proba(X_test.as_matrix())
prediction_gbm = gbm.predict(X_test.as_matrix())

gv(prediction_gbm, Y_test, prob_gbm)

###############xgboost###############

dtrain = xgb.DMatrix(X_train, Y_train)
dtest = xgb.DMatrix(X_test)
param = {'max_depth':100, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 100
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

predictions_xgb = []

for i,value in enumerate(preds):
     if value >= 0.5:
          predictions_xgb.append(1)
     else:
           predictions_xgb.append(0)

gv(predictions_xgb, Y_test, preds)

np.mean(predictions_xgb==Y_test.as_matrix().ravel())




     
