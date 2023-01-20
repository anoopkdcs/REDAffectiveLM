#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 17:55:19 2019

@author: anoop
DataSet Detals Scraper
1. Entries
2. Average Words/Sent
3. Words
"""
import glob
import string
import gensim as gs
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
import io
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Read shuffled SENh Dataset
SENHeadlines = np.load('inputs/SENrandomHeadlines.npy').tolist()  #As a list

### Sent Conter
Sent = np.zeros((0,1))


for i in range(len(SENHeadlines)):
    doc = SENHeadlines[i]
    number_of_sentences = sent_tokenize(doc)
    Sent = np.append(Sent,len(number_of_sentences))

avgposSent = np.sum(Sent)/4000
#1.1875

### word Conter
Tokens = []
totalLength = 0
tokenizer = RegexpTokenizer(r'\w+')
for i in range(len(SENHeadlines)):
    tempTokens = SENHeadlines[i].lower() #converting to lower case
    tempTokens = tempTokens.translate(str.maketrans('','',"~!@#$%^&*()_-+={}[]|\/><'?.,-+`:;"))
    tempTokens = str(" ".join([lemmatizer.lemmatize(j) for j in tempTokens.split()])) #lemmetization
    tempTokens = tokenizer.tokenize(tempTokens)
    Tokens.append(tempTokens)
    totalLength = totalLength + len(Tokens[i]) # 124172

AvgWordperDocument = totalLength/4000 #31.043


