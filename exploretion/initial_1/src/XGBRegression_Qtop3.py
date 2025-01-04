import numpy as np
import xgboost as xgb
import random
import pandas as pd
from TrAdaBoostR2_Q_env import CatalystEnvironment, QNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import copy

np.random.seed(42) 
torch.manual_seed(42)

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
    
for target in All_data_names:
        
    
    features = dfs[f'{target}'].drop(columns=['Name','ID','Yield']).values    # 特徴量
    rewards = dfs[f'{target}'][['Yield']].values   # 収率        
    
    # 出力データフレーム
    output_data = []
    
    
    for random_state in range(100):
        random.seed(random_state)
        np.random.seed(42) 
        torch.manual_seed(42)

        
        initial_indices = random.sample(range(60), 1)
        initial_features = features[initial_indices]
        initial_rewards = rewards[initial_indices]
        
        # 残りの触媒
        remaining_indices = list(set(range(60)) - set(initial_indices))
        remaining_features = features[remaining_indices]
        
        # モデルの初期化
        model = xgb.XGBRegressor(random_state=42)
        
        # モデルを訓練
        model.fit(initial_features, initial_rewards)
        
        # 選択した触媒を保存するリスト
        selected_indices = initial_indices.copy()
        
        # タイムステップごとの実際の収率と予測収率を保存するリスト
        actual_rewards_per_step = []
        predicted_rewards_per_step = []
        selected_indices_per_step = []
        
        env = CatalystEnvironment(features=features, rewards=rewards, top_k=3)
        state = env.reset()

        #最初に選んだ触媒を削除
        for index in sorted(initial_indices, reverse=True):
            env.remaining_indices.remove(index)
        
        q_network = QNetwork(input_size=33, output_size=1)
        optimizer = optim.Adam(q_network.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        gamma = 0.9  
        
        selected_features = initial_features.tolist()
        selected_rewards = initial_rewards.tolist()
        
        current_state = env.get_remaining_indices()

        # 操作を繰り返すループ
        for step in range(59):  # 全ての触媒を使うまで繰り返す
            # モデルを使って残りの触媒に対する予測を行う
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
            
            # モデルを再訓練（選ばれた触媒を追加した新しいモデルを作成）
            model.fit(selected_features, selected_rewards)
            
            current_state = copy.deepcopy(next_state)
        
            # 環境が終了したか確認
            if done:
                break
            
        
        output_data.append({'random_state': random_state,
                            'initial_indices': initial_indices, 
                            'selected_indices_per_step': selected_indices_per_step,
                            'actual_rewards_per_step':actual_rewards_per_step,
                            'predicted_rewards_per_step':predicted_rewards_per_step
                            })
        
    df = pd.DataFrame(output_data)
    df.to_csv(f'../results/XGBRegression_Qtop3_{target}_result.csv', index=False)
            
        
    
    

