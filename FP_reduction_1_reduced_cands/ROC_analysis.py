#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:27:24 2019

@author: se14
"""

# simple ROC analysis of results (not the FROC analysis!)

import pandas as pd
import sklearn.metrics
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'qt5')
matplotlib.rcParams['figure.figsize'] = (8.0, 6.0)
matplotlib.rcParams['font.size'] = 14

#%% load the results file
results_file = f'final_results/3D_CNN_FP_reduction.csv'

results_df = pd.read_csv(results_file)

labels = results_df['class']
predictions = results_df['probability']


fpr,tpr,_ = sklearn.metrics.roc_curve(labels, predictions)
auc = sklearn.metrics.roc_auc_score(labels, predictions)

plt.plot(fpr,tpr,label=f'AUC={auc}')
plt.legend()
plt.xlabel(f'FPR')
plt.ylabel(f'TPR')
plt.title(f'3D resnet false positive reduction')