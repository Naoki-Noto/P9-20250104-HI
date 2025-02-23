import numpy as np
import xgboost as xgb
import random
import pandas as pd
from TrAdaBoostR2_Q_env import CatalystEnvironment, QNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from itertools import product

# ターゲットデータの選択
target_data_names = [
    'Reaction_CO_1.5h',
    'Reaction_CO_7.5h', 
    'Reaction_CO_biphenyl', 
    'Reaction_CO_ortho', 
    'Reaction_CO_Cl', 
    'Reaction_CS', 
    'Reaction_CN', 
    'Reaction_2+2', 
    'Reaction_CF3', 
    'Reaction_CH2CF3', 
    'Reaction_CH2F', 
    'Reaction_Cy', 
    'Reaction_OCH2F', 
    'Reaction_SCF3',                 
    'Reaction_Masuda', 
    'Reaction_Sumida'
    ]

# 強化学習で選択する触媒の候補数  
top_k_list = [
    1,
    3,
    5,
    #10,
    ]

# 探索率
epsilon_list = [
    0,
    #0.05,
    #0.1,
    ]

"""

これ以降のコードは編集しない

"""

# データのインポート
All_data_names = ['Reaction_CO_1.5h', 'Reaction_CO_7.5h', 'Reaction_CO_biphenyl', 'Reaction_CO_ortho', 'Reaction_CO_Cl', 'Reaction_CS', 
                  'Reaction_CN', 'Reaction_2+2', 'Reaction_CF3', 'Reaction_CH2CF3', 'Reaction_CH2F', 'Reaction_Cy', 'Reaction_OCH2F', 'Reaction_SCF3',
                 'Reaction_Masuda', 'Reaction_Sumida']
dfs = {}

for data_name in All_data_names:
    dfs[f'{data_name}'] = pd.read_csv(f'../../data/data_60_{data_name}.csv')

    
for target, top_k, epsilon in product(
        target_data_names, top_k_list, epsilon_list):
    
    # 特徴量と収率に分ける        
    features = dfs[f'{target}'].drop(columns=['Name','ID','Yield']).values    # 特徴量
    rewards = dfs[f'{target}'][['Yield']].values   # 収率
      
    # 出力データフレーム
    output_data = []
    
    for i in range(60):
        # 順番に最初の触媒を選択
        initial_indices = [i]
        
        # seed値の固定
        random.seed(42)
        np.random.seed(42) 
        torch.manual_seed(42)
        
        # 選んだ触媒のデータを取り出す
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
        
        # 環境のリセット
        env = CatalystEnvironment(features=features, rewards=rewards, top_k=top_k)
        state = env.reset()
        
        #最初に選んだ触媒を削除
        for index in sorted(initial_indices, reverse=True):
            env.remaining_indices.remove(index)
        
        # Qネットワークの作成
        q_network = QNetwork(input_size=33, output_size=1)
        optimizer = optim.Adam(q_network.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        gamma = 0.9  
        
        # 最初の触媒の情報を保存
        selected_features = initial_features.tolist()
        selected_rewards = initial_rewards.tolist()
        
        # 現在の状態を記録
        current_state = env.get_remaining_indices()
        
        # 選択を繰り返すループ
        for step in range(59):  # 全ての触媒を使うまで繰り返す
        
            # 触媒の選択
            if random.random() < epsilon:
                # 探索: ランダムに触媒を選ぶ
                best_catalyst_remaining_index = random.choice(range(len(current_state)))
                best_catalyst_actual_index = int(current_state[best_catalyst_remaining_index])
            else:           
                # select_actionを行う
                best_catalyst_remaining_index, best_catalyst_actual_index = env.select_action(model, q_network)
            
            
            # 選択した触媒の情報を獲得
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
            
            # 現在の状態を記録
            current_state = copy.deepcopy(next_state)
            
        # 選択の過程を記録
        output_data.append({'initial_indices': initial_indices, 
                            'selected_indices_per_step': selected_indices_per_step,
                            'actual_rewards_per_step':actual_rewards_per_step,
                            'predicted_rewards_per_step':predicted_rewards_per_step
                            })
        
    df = pd.DataFrame(output_data)
    df.to_csv(f'../results/XGBRegression_Qtop{top_k}_ep{epsilon}_{target}_result.csv', index=False)
    
    
    
