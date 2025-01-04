# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:08:15 2024

@author: noyor
"""

import pandas as pd
from adapt.instance_based import TrAdaBoostR2
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor as lgbm
from sklearn.ensemble import HistGradientBoostingRegressor as hgb
from sklearn.ensemble import RandomForestRegressor as rfr
import random
from TrAdaBoostR2_Q_env import CatalystEnvironment, QNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from Dataset_Selection import DatasetSelect

# random_stateの固定
np.random.seed(42) 
torch.manual_seed(42)

# dataのインポート
All_data_names = [#'Reaction_CO_1.5h', 'Reaction_CO_7.5h', 'Reaction_CO_biphenyl', 'Reaction_CO_ortho',
                  'Reaction_CO_Cl',
                  #'Reaction_CS', 
                  #'Reaction_CN',
                  'Reaction_2+2',
                  #'Reaction_CF3', 'Reaction_CH2CF3', 'Reaction_CH2F', 'Reaction_Cy', 'Reaction_OCH2F', 'Reaction_SCF3',
                 #'Reaction_Masuda', 'Reaction_Sumida'
                 ]

dfs = {}

for data_name in All_data_names:
    dfs[f'{data_name}'] = pd.read_csv(f'../../data/data_60_{data_name}.csv')

selection = DatasetSelect()

for data_name in All_data_names:
    selection.import_dataset(dfs[f'{data_name}'], data_name)
    
#TrAdaBoostR2の学習器の設定
es_dict = {
           #"XGB": XGBRegressor(random_state=42), 
           "LGBM": lgbm(random_state=42), 
           }
ne = 5

# souceの数
k = 5

ep_list = [0.05, 0.07, 0.1]

# top ot bottom
selection_order_list = [
    "top",
    #"bottom",
    ]

    
for target_data_name in All_data_names:
    
    for selection_order in selection_order_list:
    
        for es_name, es in es_dict.items():
            
            for epsilon in ep_list:
                
                # target＿dataとsource_data候補を分離する
                All_source_data_names = [item for item in All_data_names if item != target_data_name]
                
                # 出力データフレーム
                output_data = []
                
                for random_state in range(100):
                    random.seed(random_state)
                    np.random.seed(42) 
                    torch.manual_seed(42)
                    
                    #　最初の触媒を選択
                    initial_indices = random.sample(range(60), 1)
                    
                    # 相関係数を計算しｓource_dataを選択する
                    selection.calculate_corr(selected_catalyst_index=initial_indices)
                    
                    source_data_names = selection.select_source_data(
                        target_data_name=target_data_name,
                        source_data_candidates=All_source_data_names,
                        selection_order=selection_order,
                        k=k
                        )
                    
                    Xs, ys, Xt, yt = selection.shaping_df(
                        source_data_names=source_data_names,
                        target_data_name=target_data_name
                        )
                    
                    # 選んだ触媒のXｙを取ってきてmodel fitをする
                    initial_Xt = Xt.iloc[initial_indices]
                    initial_yt = yt.iloc[initial_indices]
                    
                    model = TrAdaBoostR2(es, n_estimators=ne, Xt=initial_Xt, yt=initial_yt, random_state=42)
                    model.fit(Xs, ys)
                    
                    Xs_np, ys_np, Xt_np, yt_np, initial_Xt_np, initial_yt_np = (df.to_numpy() for df in [Xs, ys, Xt, yt, initial_Xt, initial_yt])
                    
                    env = CatalystEnvironment(features=Xt_np, rewards=yt_np, top_k=60)
                    state = env.reset()
                    
                    #最初に選んだ触媒を削除
                    for index in sorted(initial_indices, reverse=True):
                        env.remaining_indices.remove(index)
                    
                    q_network = QNetwork(input_size=19, output_size=1)
                    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
                    criterion = nn.MSELoss()
                    gamma = 0.9  
                    
                    # モデル更新用に選択した特徴量と収率を保持するリスト
                    selected_features = initial_Xt_np.tolist()
                    selected_rewards = initial_yt_np.tolist()
                    
                    # タイムステップごとの実際の収率と予測収率を保存するリスト
                    actual_rewards_per_step = []
                    predicted_rewards_per_step = []
                    selected_indices_per_step = []
                    
                    current_state = env.get_remaining_indices()
                    
                    for step in range(59):
                        
                        if random.random() < epsilon:
                            # 探索: ランダムに触媒を選ぶ
                            best_catalyst_remaining_index = random.choice(range(len(current_state)))
                            best_catalyst_actual_index = int(current_state[best_catalyst_remaining_index])
                        else:
                            # select_actionを行う
                            best_catalyst_remaining_index, best_catalyst_actual_index = env.select_action(model, q_network)
                    
                        
                        # 選択した触媒に基づいて予測収率を計算
                        best_catalyst_feature = env.get_remaining_features()[best_catalyst_remaining_index]
                        best_catalyst_feature = np.array(best_catalyst_feature).reshape(1, -1)  # 1次元ベクトルを2次元に変換
                        predicted_reward = model.predict(best_catalyst_feature).item()  # 予測収率
                        
                        # 環境に正しいインデックスを与える
                        next_state, actual_reward, done = env.step(best_catalyst_actual_index)
                    
                        # タイムステップごとの実際の収率と予測収率を記録
                        actual_rewards_per_step.append(actual_reward)
                        predicted_rewards_per_step.append(predicted_reward)
                        selected_indices_per_step.append(best_catalyst_actual_index)
                    
                        #Q値の更新のために状態をテンソルにする
                        state_tensor = torch.tensor(env.features[current_state], dtype=torch.float32)
                        next_state_tensor = torch.tensor(env.features[next_state], dtype=torch.float32)
                    
                        # 損失の計算
                        loss = q_network.update_q_values(
                        optimizer=optimizer,
                        criterion=criterion,
                        state_tensor=state_tensor,
                        actual_reward=actual_reward,
                        predicted_reward=predicted_reward,
                        action=best_catalyst_remaining_index,
                        gamma=gamma,
                        next_state_tensor=next_state_tensor
                        )
                    
                        # 選択した触媒をリストに追加
                        selected_features.append(best_catalyst_feature[0].tolist())
                        selected_rewards.append(actual_reward.tolist())
                        
                        # 選択した触媒を含めて相関係数を再計算
                        selected_indices = initial_indices + selected_indices_per_step
                        selection.calculate_corr(selected_catalyst_index=selected_indices)
                        
                        # source＿dataの選択
                        source_data_names = selection.select_source_data(
                            target_data_name=target_data_name,
                            source_data_candidates=All_source_data_names,
                            selection_order=selection_order,
                            k=k
                            )
                        Xs, ys, Xt, yt = selection.shaping_df(
                            source_data_names=source_data_names,
                            target_data_name=target_data_name
                            )
                    
                        # 選択した触媒を使ってモデルを再訓練
                        selected_features_df = pd.DataFrame(selected_features)
                        selected_rewards_df = pd.DataFrame(selected_rewards)
                        model = TrAdaBoostR2(es, n_estimators=ne, Xt=selected_features_df, yt= selected_rewards_df, random_state=42)
                        model.fit(Xs, ys) 
                    
                        current_state = copy.deepcopy(next_state)
                    
                        # 環境が終了したか確認
                        if done:
                            break
                    
                    # 記録用リストに追加
                    output_data.append({'random_state': random_state,
                                        'initial_indices': initial_indices, 
                                        'selected_indices_per_step': selected_indices_per_step,
                                        'actual_rewards_per_step': actual_rewards_per_step,
                                        'predicted_rewards_per_step': predicted_rewards_per_step,
                                        })
                # 各反応ごとに保存    
                df = pd.DataFrame(output_data)
                df.to_csv(f'../results/TrAdaBoostR2_QAll_ep{epsilon}_{es_name}_ne{ne}_{target_data_name}_source{selection_order}5_result.csv', index=False)
                
            
    
      