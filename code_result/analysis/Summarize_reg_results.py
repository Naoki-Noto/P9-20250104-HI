# -*- coding: utf-8 -*-
"""
Created on Tue May 13 09:45:07 2025

@author: noyor
"""

import pandas as pd
import numpy as np

model_list = ['RF', 'TrAB']

train_size_list = [15, 20, 30]

All_data_names = ['Reaction_CO_1.5h', 'Reaction_CO_biphenyl', 'Reaction_CO_ortho', 'Reaction_CO_Cl',
                  'Reaction_CS', 'Reaction_CN', 'Reaction_2+2', 
                  'Reaction_CF3', 'Reaction_CH2CF3', 'Reaction_CH2F', 'Reaction_Cy', 'Reaction_SCF3',
                  'Reaction_OCH2F', 'Reaction_P', 'Reaction_Si']

for model in model_list:
    results = []

    for target in All_data_names:
        for train_size in train_size_list:
            df = pd.read_csv(f'../results_reg/{model}/train_size_{train_size}/result_{target}.csv')
            results.append({
                'model': model,
                'reaction': target,
                'train_size': train_size,
                'Avg_R2': np.mean(df['r2_test']),
                'Max_R2': np.max(df['r2_test']),
                'Std_R2': np.std(df['r2_test'], ddof=0),
                'Avg_PearsonR': np.mean(df['pr_test']),
                'Max_PearsonR': np.max(df['pr_test']),
                'Std_PearsonR': np.std(df['pr_test'], ddof=0),
                'Avg_kendalltau': np.mean(df['kt_test']),
                'Max_kendalltau': np.max(df['kt_test']),
                'Std_kendalltau': np.std(df['kt_test'], ddof=0),
                })

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(f'./results/summary_reg/summary_{model}_trainsize_results.csv', index=False)
    