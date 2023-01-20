#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:48:44 2020

@author: anoop

Although we have no means to verify the actual absolute number
of votes collected by the Mood Meter, we can provide a very conservative 
estimate for it: by computing the lower common denominator over the percentages 
of affective votes obtained by a Rappler article, we can derive the minimum 
number of votes needed to obtain them

Ref: Deep Feelings: A Massive Cross-Lingual Study on the
Relation between Emotions and Viralitys
"""
import numpy as np
from fractions import Fraction

SENlabelsB4Normalization = np.load('inputs/SENLabelsB4Normalization.npy') # As a multi dimensional array
np.argwhere(np.isnan(SENlabelsB4Normalization)) #is used to find the NAN in the matrix


SENlabelsB4Normalization = np.int32(SENlabelsB4Normalization)

# total votes for a particular news
rowSum = np.sum(SENlabelsB4Normalization, axis = 1 ) 

# Delete zero entries in the rowSum to avoid division by zero
for i in range (0,np.size(SENlabelsB4Normalization,0)):
    if rowSum[i] == 0:
        print('delete '+str(i))
#rowSum = np.delete(rowSum,1754,0) 
#SENlabelsB4Normalization = np.delete(SENlabelsB4Normalization,1754,0)

# Simplify fraction and find Denominator row-wise, i.e.; for each news
# Fraction: (vote for each emotion in a particualr news)/(total votes for that news)
SENlabelsB4Normalization_denom = np.zeros((np.size(SENlabelsB4Normalization,0),np.size(SENlabelsB4Normalization,1)))
for i in range (0,np.size(SENlabelsB4Normalization,0)):
    for j in range (0,np.size(SENlabelsB4Normalization,1)):
        temp = Fraction(SENlabelsB4Normalization[i,j], rowSum[i]) # simplify the fraction
        SENlabelsB4Normalization_denom[i,j] = temp.denominator # find denominator of simplified fraction

SENlabelsB4Normalization_lcd = np.zeros((np.size(SENlabelsB4Normalization,0),1))
for k in range (0,np.size(SENlabelsB4Normalization,0)):
    temp_list = np.int64(SENlabelsB4Normalization_denom[k]).tolist()
    SENlabelsB4Normalization_lcd[k] = np.lcm.reduce(temp_list) # calculate lcm of the denominators, row-wise

min_voters = np.sum(SENlabelsB4Normalization_lcd) 
#242680.0