# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:06:37 2017

@author: Sudipta
"""

from __future__ import division
import csv
import pip
from gensim import corpora, models, similarities
from sklearn import model_selection
from sklearn import datasets, metrics, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
train_file = "train.csv"
test_file = "test.csv"
df = pd.read_csv(train_file, index_col="id")
df_test = pd.read_csv(test_file)
#print(df_test.iloc[:,1:2])
#df = [df_train, df_test]
SUBMIT_PATH = 'rf_submission_2.csv'
print(df.head())

#import matplotlib.pylab as plt
questions = dict()

for row in df.iterrows():
    questions[row[1]['qid1']] = row[1]['question1']
    questions[row[1]['qid2']] = row[1]['question2']
    
import re
import nltk
#nltk.download()
def basic_cleaning(st):
    st = repr(st)
    try:
        st = st.decode('unicode-escape')
    except Exception:
        pass
    st = st.lower()
    st = re.sub(' +', ' ', st)
    return st

sentences = []
for i in questions:
    sentences.append(nltk.word_tokenize(basic_cleaning(questions[i])))
    
import gensim
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

tf = dict()
docf = dict()
total_docs = 0
for qid in questions:
    total_docs += 1
    toks = nltk.word_tokenize(basic_cleaning(questions[qid]))
    uniq_toks = set(toks)
    for i in toks:
        if i not in tf:
            tf[i] = 1
        else:
            tf[i] += 1
    for i in uniq_toks:
        if i not in docf:
            docf[i] = 1
        else:
            docf[i] += 1
                

import math
def idf(word):
    return 1 - math.sqrt(docf[word]/total_docs)

print(idf("nightmare"))

#import re
import nltk
def basic_cleaning(string):
    string = repr(string)
    string = string.lower()
    string = re.sub('[0-9\(\)\!\^\%\$\'\"\.;,-\?\{\}\[\]\\/]', ' ', string)
    string = ' '.join([i for i in string.split() if i not in ["a", "and", "of", "the", "to", "on", "in", "at", "is"]])
    string = re.sub(' +', ' ', string)
    return string

def w2v_sim(w1, w2):
    try:
        return model.similarity(w1, w2)*idf(w1)*idf(w2)
    except Exception:
        return 0.0

def img_feature(row):
    s1 = row['question1']
    s2 = row['question2']
    t1 = list((basic_cleaning(s1)).split())
    t2 = list((basic_cleaning(s2)).split())
    Z = [[w2v_sim(x, y) for x in t1] for y in t2] 
    a = np.array(Z, order='C')
    return [np.resize(a,(10,10)).flatten()]
s = df

img = s.apply(img_feature, axis=1, raw=True)
pix_col = [[] for y in range(100)] 
for k in img.iteritems():
        for f in range(len(list(k[1][0]))):
           pix_col[f].append(k[1][0][f])

    
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in repr(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in repr(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

train_word_match = df.apply(word_match_share, axis=1, raw=True)

x_train = pd.DataFrame()

for g in range(len(pix_col)):
    x_train['img'+repr(g)] = pix_col[g]
  
#x_train['word_match'] = train_word_match

y_train = s['is_duplicate'].values           
  
pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]
# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

del pos_train, neg_train



#from sklearn.cross_validation import train_test_split
#from sklearn.ensemble import RandomForestClassifier

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
#clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf = GradientBoostingRegressor(n_estimators=2000, random_state=1)
clf.fit(x_train,y_train)
#clf = LogisticRegression()
#clf.fit(x_train, y_train)
'''clf.predict(x_valid)
print(clf.score(x_train, y_train))
print(clf.score(x_valid, y_valid))'''

  
print('test started')
questions_test = dict()
#print(df_test)
i=1
for row in df_test.iterrows():
    #print(1);
    questions_test[i] = row[1]['question1']
    i+=1
    questions_test[i] = row[1]['question2']

print('dictionary created')
  
sentences_test = []
for i in questions_test:
    sentences_test.append(nltk.word_tokenize(basic_cleaning(questions_test[i])))

print('All sentenses are tokensized')

model_test = gensim.models.Word2Vec(sentences_test, size=100, window=5, min_count=5, workers=4)

print('Test model created bu gensim')

tf_test = dict()
docf_test = dict()
total_docs = 0
for qid in questions_test:
    total_docs += 1
    toks = nltk.word_tokenize(basic_cleaning(questions_test[qid]))
    uniq_toks = set(toks)
    for i in toks:
        if i not in tf_test:
            tf_test[i] = 1
        else:
            tf_test[i] += 1
    for i in uniq_toks:
        if i not in docf_test:
            docf_test[i] = 1
        else:
            docf_test[i] += 1
                     
def img_feature1(row):    
    s1 = row[0]
    s2 = row[1]    
    t1 = list((basic_cleaning(s1)).split())
    t2 = list((basic_cleaning(s2)).split())
    #print('printing row t1 - > ', t1)
    #print('printing row t2 - > ', t2)
    Z = [[w2v_sim(x, y) for x in t1] for y in t2] 
    a = np.array(Z, order='C')
    return [np.resize(a,(10,10)).flatten()]  
                
s_test = df_test.iloc[:,1:]

img_test = s_test.apply(img_feature1, axis=1, raw=True)
pix_col_test = [[] for y in range(100)] 
for k in img_test.iteritems():
        for f in range(len(list(k[1][0]))):
           pix_col_test[f].append(k[1][0][f])
           
x_test = pd.DataFrame()

for g in range(len(pix_col_test)):
    x_test['img'+repr(g)] = pix_col_test[g]
    
print('predicting test started')
  
'''
df = reg.transform(test[['question1','question2']])
sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = lr.predict(df.toarray())
sub.to_csv('latest_trial.csv', index=False)'''
  
test_prediction = clf.predict(x_test)

df_test["is_duplicate"] = test_prediction
df_test[['test_id','is_duplicate']].to_csv(SUBMIT_PATH, header=True, index=False)
#df_test["is_duplicate"].to_csv(SUBMIT_PATH, header=True, index=False)
print("Done!")
#for i in test_prediction:
#    print(i)
    

