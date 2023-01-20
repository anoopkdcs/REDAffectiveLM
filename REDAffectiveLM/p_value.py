# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from sklearn import metrics


#load data our model
rmse_all_vals_our_model = np.load('file name.npy')
rmse_all_vals_our_model = np.reshape(rmse_all_vals_our_model,(len(rmse_all_vals_our_model)))
acc1_all_vals_our_model =np.load('file name.npy')
acc1_all_vals_our_model = np.reshape(acc1_all_vals_our_model,(len(acc1_all_vals_our_model)))

#load data best baseline SSBED
rmse_all_vals_ssbed = np.load('file name.npy')
rmse_all_vals_ssbed = np.reshape(rmse_all_vals_ssbed, (len(rmse_all_vals_ssbed)))
acc1_all_vals_ssbed =np.load('file name.npy')
acc1_all_vals_ssbed = np.reshape(acc1_all_vals_ssbed,(len(acc1_all_vals_ssbed)))

#McNemar’s Test  for ACC@1 
#https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
contingency_table = metrics.confusion_matrix(acc1_all_vals_our_model, acc1_all_vals_ssbed,labels=[1,0])
pval = mcnemar(contingency_table, exact=True)
print(pval)

#Student t-test for RMSE
#https://machinelearningmastery.com/use-statistical-significance-tests-interpret-machine-learning-results/
value, pvalue = ttest_ind(rmse_all_vals_our_model, rmse_all_vals_ssbed, equal_var=False)
print(value, pvalue)

#Kolmogorov-Smirnov test for RMSE
#https://machinelearningmastery.com/use-statistical-significance-tests-interpret-machine-learning-results/
value, pvalue = ks_2samp(rmse_all_vals_our_model, rmse_all_vals_ssbed)
print(value, pvalue)