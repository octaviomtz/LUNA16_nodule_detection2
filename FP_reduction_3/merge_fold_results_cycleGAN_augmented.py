#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:44:14 2019

@author: se14
"""
import pandas as pd
import os
# script to merge results for all folds

print('Merging the results from all folds')

out_foldr = 'augmented/final_results/'
if (not os.path.exists(out_foldr)) & (out_foldr != ""): 
    os.makedirs(out_foldr)

out_filename = 'multilevel_3D_CNN_FP_reduction.csv'
all_results_df = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','probability'])

for fold_k in range(10):
    
    try:
        results_filename = f'test_results_fold{fold_k}.csv'
        out_path = f'augmented/results_fold_{fold_k}/'
        results_filename = out_path + results_filename
        
        tmpDf = pd.read_csv(results_filename)
        
        all_results_df = all_results_df.append(tmpDf,ignore_index=False,sort=False).reset_index(drop=True)
    
    except:
        print('Error')
        continue

all_results_df.to_csv(out_foldr + out_filename,index=False)
    
#%%