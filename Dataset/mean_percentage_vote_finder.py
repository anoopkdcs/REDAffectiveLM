#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:31:03 2020

@author: Anoop
mean percentage of votes for each emotional dimension 
"""
import numpy as np

SENlabelsNormalized = np.load('inputs/SENrandomLabels.npy') # As a multi dimensional array
#np.argwhere(np.isnan(SENlabelsNormalized))
meanPercVotesNormalized = np.mean(SENlabelsNormalized, axis = 0)
#joy = 0.3137
#Sadness = 0.0781
#anger = 0.3388
#fear = 0.1475
#surprise = 0.1218

