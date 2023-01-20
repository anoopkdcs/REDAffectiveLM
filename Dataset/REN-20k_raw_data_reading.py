#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 10:24:18 2021
@author: Anoop
Read REN20k Long text
"""

import pandas as pd  
import glob
import numpy as np

SENnewsdocs = []
SENlabelsList = []
files = glob.glob('inputs/*.xlsx') # read all .xlsx files in the directory 

for f in range(len(files)): # loop helps to read all files one by one 
    SENDataSheet = pd.read_excel(files[f])
    print(files[f])
    valuesSENDataSheet=SENDataSheet.values
    newsdocs = []
    labels = []
    titleTemp = ""
    abstractTemp = ""
    joyTemp = 0.0 
    sadTemp = 0.0
    angerTemp = 0.0
    fearTemp = 0.0
    surpriseTemp = 0.0
    newsdocTemp = ""
    labelTemp = []    
        
    for i in range(len(valuesSENDataSheet)): # loop helps to handle news documents and emotions in a single .xlxs file.
        titleTemp = str(valuesSENDataSheet[i][0]) # news title 
        abstractTemp = str(valuesSENDataSheet[i][1]) #news abstract 
        contentTemp = str(valuesSENDataSheet[i][2]) #news content 
        joyTemp = float(valuesSENDataSheet[i][3]) #joy
        sadTemp = float(valuesSENDataSheet[i][4]) #Sadness
        angerTemp = float(valuesSENDataSheet[i][5]) #anger
        fearTemp = float(valuesSENDataSheet[i][6]) #fear
        surpriseTemp = float(valuesSENDataSheet[i][7]) #surprise
        
        #newsdocTemp = titleTemp +" "+ abstractTemp #compaining title and abstract of news 
        newsdocTemp = titleTemp +" "+ abstractTemp + " "+ contentTemp #compaining title, abstract, and the content of news 
        labelTemp = [angerTemp,fearTemp, joyTemp,  sadTemp, surpriseTemp  ] #only selecting the ekman emotions
        
        newsdocs.append(newsdocTemp)
        labels.append(labelTemp)
        
    SENnewsdocs = SENnewsdocs + newsdocs #appending news documents from multiple xlxs files 
    SENlabelsList = SENlabelsList +labels #appending labels from multiple xlxs files 
    SENlabels  = np.asarray(SENlabelsList)

#Check any NaN value present 
array_sum = np.sum(SENlabels)
array_has_nan = np.isnan(array_sum)
print(array_has_nan) #if False then no NaN
   
#Check any row with full zeros
print(np.where(~SENlabels.any(axis=1))[0])
#output: [ ]

#checking values > 1
grater = np.argwhere(SENlabels > 1)
print(grater)
#[]

#delete the data which have index > 1s
#SENnewsdocs = np.delete(SENnewsdocs, 7068, 0)
#SENlabels = np.delete(SENlabels, 7068, 0)

# Save news and coresponding Labels 
np.save('outputs/REN-10k_extended_headline_abstract_content.npy',SENnewsdocs)
np.save('outputs/REN-10k_extended_headline_abstract_labels.npy', SENlabels) 

# load to another program 
#SENnewsdocsNormalized = np.load('outputs/SENnewsdocsNormalized.npy').tolist()  #As a list
#SENnewsdocsNormalized = np.load('outputs/SENnewsdocsNormalized.npy') # As a multi dimensional array