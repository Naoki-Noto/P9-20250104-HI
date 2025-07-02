# -*- coding: utf-8 -*-
"""
Created on Wed May  7 21:18:37 2025

@author: noyor
"""

import pandas as pd
import ast

model_list = [
    "RFsearch",
    "TrABsearch_top5_e0_ne5",
    "TrABsearch_top3_e0_ne5",
    "TrABsearch_all_e0_ne5",
    "BOsearch_EI",
    "BOsearch_PI",
    "BOsearch_UCB",
    ]

for model in model_list:
    if model == 'TrABsearch_top5_e0_ne5':
        file_name = 'TrABsearch_Reaction_CO_biphenyl_top5_e0_ne5_result'
    elif model == 'TrABsearch_top3_e0_ne5':
        file_name = 'TrABsearch_Reaction_CO_biphenyl_top3_e0_ne5_result'
    elif model == 'TrABsearch_all_e0_ne5':
        file_name = 'TrABsearch_Reaction_CO_biphenyl_top14_e0_ne5_result'
    elif model == 'RFsearch':
        file_name = 'RFsearch_Reaction_CO_biphenyl_result'
    elif model == 'BOsearch_EI':
        file_name = 'BOsearch_Reaction_CO_biphenyl_result'
    elif model == 'BOsearch_PI':
        file_name = 'BOsearch_Reaction_CO_biphenyl_result'
    elif model == 'BOsearch_UCB':
        file_name = 'BOsearch_Reaction_CO_biphenyl_result'
    else:
        print(f'Not acceptable: {model}')
        continue
    
    df = pd.read_csv(f'../results_Oos/{model}/{file_name}.csv')
    df['selected_indices_per_step'] = df['selected_indices_per_step'].apply(ast.literal_eval)
    
    summary_df = pd.DataFrame()
    summary_df['total_selected_OPS'] = df['selected_indices_per_step'].apply(len)
    summary_df.insert(0, 'Run_index', range(60))
    summary_df.to_csv(f'./results/summary_OoS/{model}/{file_name}_OoS_summary.csv', index=False)
    
    mean_series = summary_df.drop(columns='Run_index').mean()
    std_series = summary_df.drop(columns='Run_index').std(ddof=0)
    columns = mean_series.index
    average_df = pd.DataFrame([mean_series.values], columns=columns)
    std_df = pd.DataFrame([std_series.values], columns=columns)
    summary_stats_df = pd.concat([average_df, std_df], ignore_index=True)
    summary_stats_df.insert(0, 'Stat', ['average', 'std'])
    summary_stats_df.to_csv(f'./results/summary_OoS/{model}/{file_name}_OoS_stats.csv', index=False)

