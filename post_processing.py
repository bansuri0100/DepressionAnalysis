#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np
import spacy
nlp=spacy.load('en_core_web_sm')
import nltk


# In[2]:


df4=pd.read_csv('cleaned_data.csv')


# In[3]:


from sklearn.model_selection import train_test_split



X = df4['cleaned_sentence']
y = df4['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


# In[4]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])

# Feed the training data through the pipeline
text_clf.fit(X_train.values.astype('U'), y_train) 


# In[5]:


def process(str):
    corpus=[]
    
    sentence=re.sub('[^a-zA-Z]', ' ',str)
    sentence=sentence.lower()
    sentence=sentence.split()
    
    sentence=[s for s in sentence if not nlp.vocab[s].is_stop]
    sentence=' '.join(sentence)
    
    sent=nlp(sentence)   
    sent2=[s.lemma_ for s in sent ]
    sentence2=' '.join(sent2)
    return(sentence2)


# In[20]:



    string=str(input("Enter Message :"))
    string2=process(string)   
    z=pd.Series(string2)
    predictions = text_clf.predict(z)
    predictions
   

