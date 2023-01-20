#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:22:32 2019
@author: Anoop
Lexicon Coverage finder NRC Emolex
"""
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import csv
from collections import Counter

#load Data and Label
SENh = np.load('inputs/name_of_input_textdata_file.npy').tolist()
dataSet_1000 = SENh

#final  headlines, tokenized 
headline_tokens = [] 
for i in range(len(dataSet_1000)):
    temp_headline_tokens = dataSet_1000[i].lower() #converting to lower case
    temp_headline_tokens = temp_headline_tokens.translate(str.maketrans('','',"~!@#$%^&*()_-+={}[]|\/><'?.,-+`:;"))
    temp_headline_tokens = str(" ".join([lemmatizer.lemmatize(j) for j in temp_headline_tokens.split()])) #lemmetization
    temp_headline_tokens = nltk.word_tokenize(temp_headline_tokens)
    headline_tokens.append(temp_headline_tokens)
    
#total word tockens befor POS removal 
totalTokens = 0
for r in range(len(headline_tokens)):
    totalTokens = totalTokens + len(headline_tokens[r])
    print(totalTokens)
    
# Removal of word tokens except noun, verb, adjective, adverb
finalHeadlines = []
for t in range(len(headline_tokens)):
    pos = nltk.pos_tag(headline_tokens[t])
    tempTocken = []
    for p in range(len(pos)):
        if (pos[p][1] == 'NN') or (pos[p][1] == 'VB') or (pos[p][1] == 'JJ') or (pos[p][1] == 'RB'):
            tempTocken.append(pos[p][0])
    finalHeadlines.append(tempTocken)
    
#total word tockens after POS removal 
totalTokensAfPOS = 0
for ft in range(len(finalHeadlines)):
    totalTokensAfPOS = totalTokensAfPOS + len(finalHeadlines[ft])
    print(totalTokensAfPOS)
    
    
#EMolex Reading 
Path_intensityLex= "Lexicons/emoWordnet.csv"
file = open(Path_intensityLex,'r')
reader = csv.reader(file, delimiter=',')
emotionList = []
for csvrow in reader:
    key = csvrow[0]
    emotionList.append(key)

#Read unique words & adding into a list 
count = Counter(emotionList)
lexWordList = []
for key, value in count.items():
    if key.isupper(): # TRUE is in Upper case, So need to convert into lower case 
        print(key)
        key = key.lower()
        key = lemmatizer.lemmatize(key)
        print(key)
    key = lemmatizer.lemmatize(key)
    lexWordList.append(key)
    

#Creating Sigle list of Corpus
coupus_list = []
for l in range(len(finalHeadlines)):
    coupus_list = coupus_list + finalHeadlines[l]

#unique words in corpus 
    
coun_corpus = Counter(coupus_list)
coupus_wordList = []
for c_key, c_value in coun_corpus.items():
    coupus_wordList.append(c_key)

#count common elements
common_elem_count = len(set(lexWordList)&set(coupus_wordList))
        
lex_coverage_by_emolex = common_elem_count / len(lexWordList)
lex_coverage_by_corpus = common_elem_count / len(coupus_wordList)

print("lex_coverage_by_emolex: " + str(lex_coverage_by_emolex))
print("lex_coverage_by_corpus: " + str(lex_coverage_by_corpus))


        
    
   

