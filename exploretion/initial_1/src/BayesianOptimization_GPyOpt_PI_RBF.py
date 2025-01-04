# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:48:42 2024

@author: noyor
"""

import pandas as pd
import GPy
import GPyOpt
import numpy as np
import random


# データの準備
All_data_names = ['Reaction_CO_1.5h', 'Reaction_CO_7.5h', 'Reaction_CO_biphenyl', 'Reaction_CO_ortho', 'Reaction_CO_Cl', 'Reaction_CS', 
                  'Reaction_CN', 'Reaction_2+2', 'Reaction_CF3', 'Reaction_CH2CF3', 'Reaction_CH2F', 'Reaction_Cy', 'Reaction_OCH2F', 'Reaction_SCF3',
                 'Reaction_Masuda', 'Reaction_Sumida']

dfs = {}

for data_name in All_data_names:
    dfs[f'{data_name}'] = pd.read_csv(f'../../data/data_60_{data_name}.csv')

for target in All_data_names:    

    # 報酬関数の定義
    def reward_function(x):
        index = int(x[0, 0])
        return dfs[f'{target}'].loc[index, 'Yield']  
    
    # 探索範囲の定義
    bounds = [{'name': 'index', 'type': 'discrete', 'domain': np.arange(60)}]
    
    # kernel
    kernel = GPy.kern.RBF(input_dim=1, lengthscale=1.0)
    
    # 出力データフレーム
    output_data = []
    
    for random_state in range(100):
        random.seed(random_state)
        
        initial_indices = random.sample(range(60), 1)
        initial_indices = np.array(initial_indices)
        
        initial_X = initial_indices[:, np.newaxis]
        initial_y = dfs[target].iloc[initial_indices]['Yield'].values.reshape(-1, 1)
        
        # 選択されたインデックスの記録
        selected_indices = list(initial_indices)
        remaining_indices = np.setdiff1d(np.arange(60), selected_indices)
        
        # output用リスト
        selected_indices_per_step = []
        actual_rewards_per_step = []
        predicted_rewards_per_step = []
        
        opt = GPyOpt.methods.BayesianOptimization(
            f=reward_function,          # 最適化する関数（報酬関数）
            domain=[{'name': 'index', 'type': 'discrete', 'domain': remaining_indices}],
            kernel=kernel,
            X=initial_X,                # 最初の5つのインデックス
            Y=initial_y,                # 最初の5つの報酬
            acquisition_type='MPI',
            acquisition_weight=1.0,
            exact_feval=True,           # 評価関数が正確であると仮定
            maximize=True               # 報酬を最大化するタスク
        )
        
        
        # 最適化の実行
        for _ in range(59):  
                
            opt.run_optimization(max_iter=1)  # 1つずつ最適化を進める
        
            # 新しく選ばれたインデックスを記録
            new_index = int(opt.X[-1][0])  # 直近で選ばれたインデックスを取得
            selected_indices_per_step.append(new_index)
            selected_indices.append(new_index) 
            
            # 新しく選ばれた触媒の収率を記録
            actual_rewards_per_step.append(dfs[target].loc[new_index, 'Yield'])
        
            # 探索範囲から新しく選ばれたインデックスを除外
            remaining_indices = np.setdiff1d(np.arange(60), selected_indices)
     
            if len(remaining_indices) == 0:
                print(f"All indices selected for random_state {random_state}. Stopping optimization.")
                break  # 残りのインデックスがなくなった場合、ループを終了        
            
            
            opt = GPyOpt.methods.BayesianOptimization(
                f=reward_function,
                domain=[{'name': 'index', 'type': 'discrete', 'domain': remaining_indices}],
                kernel=kernel,
                X=opt.X,               # これまでの探索結果を引き継ぐ
                Y=opt.Y,               # これまでの報酬を引き継ぐ
                acquisition_type='MPI',
                acquisition_weight=1.0,
                exact_feval=True,
                maximize=True
            )
        
        
        # 記録    
        output_data.append({'random_state': random_state,
                            'initial_indices': initial_indices, 
                            'selected_indices_per_step': selected_indices_per_step,
                            'actual_rewards_per_step':actual_rewards_per_step,
                            'predicted_rewards_per_step':predicted_rewards_per_step
                            })
        
    
    
    # csvファイルで出力    
    df = pd.DataFrame(output_data)
    df.to_csv(f'../results/BayesianOptimization_{target}_GPyOpt_PI_RBF_result.csv', index=False)

