# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:28:25 2024

@author: noyor
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class CatalystEnvironment:
    def __init__(self, features, rewards, top_k):
        self.features = features  
        self.rewards = rewards   
        self.remaining_indices = list(range(len(features)))  
        self.top_k = top_k       
        self.done = False
        self.count = 0         
        
    def reset(self):
        """ 環境のリセット。すべての触媒が未選択の状態に戻る """
        self.remaining_indices = list(range(len(self.features)))
        self.done = False
        return self.remaining_indices
    
    def select_Q_action(self, q_network):
        # remaining_indicesの確認
        remaining_indices = self.remaining_indices
        with torch.no_grad():
            state_tensor = torch.FloatTensor(self.get_remaining_features())  
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item() # 選択した触媒のremaining_indices内のインデックスをスカラーで出力
            
        
         # 選択した触媒のインデックスを得る
            best_catalyst_remaining_index = action
            best_catalyst_actual_index = int(remaining_indices[best_catalyst_remaining_index])

        return best_catalyst_remaining_index, best_catalyst_actual_index
        

    def select_action(self, model, q_network):
        """予測収率の上位k個の触媒のQ値を求め、最大の触媒のインデックスを返す。"""

        # 残っている触媒の予測収率を取得
        predicted_rewards = model.predict(pd.DataFrame(self.get_remaining_features()))
         # predicted_rewards を 1次元配列に変換
        predicted_rewards_1d = predicted_rewards.ravel()

        if self.count == 0:  # 1手目の場合
               # 収率予測が最も高い触媒のインデックスを取得
               best_catalyst_remaining_index = np.argmax(predicted_rewards_1d)
               best_catalyst_actual_index = self.remaining_indices[best_catalyst_remaining_index]
               
        else: # 2手目以降
            k = min(self.top_k, len(self.remaining_indices))
            # 各ステップごとに `top_k_indices` をリセットして再計算
            top_k_indices_in_remaining = np.argsort(predicted_rewards_1d)[-k:]
    
            # `top_k_indices` は `remaining_indices` に基づくインデックスを返す必要がある
            top_k_indices = [self.remaining_indices[i] for i in top_k_indices_in_remaining]
          
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.get_remaining_features()[top_k_indices_in_remaining])  # 上位k個を入力
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values).item()  # 選択した触媒のtop_k_indices内のインデックスをスカラーで出力
            
             # 選択した触媒のインデックスを得る
                best_catalyst_index_in_top_k = action
                best_catalyst_remaining_index = int(top_k_indices_in_remaining[best_catalyst_index_in_top_k])
                best_catalyst_actual_index = int(top_k_indices[best_catalyst_index_in_top_k])

        self.count += 1

        return best_catalyst_remaining_index, best_catalyst_actual_index

    
    def step(self, chosen_index):
        """
        行動（action）として与えられた触媒のインデックスに基づき、次の状態と報酬を返す。
        chosen_index: 残っている触媒のインデックス
        """
 
        # 実際の収率を取得
        actual_reward = self.rewards[chosen_index]

        # 選ばれた触媒を残りから除外
        self.remaining_indices.remove(chosen_index)

        # 状態遷移（次の状態）
        next_state = self.remaining_indices
    
        # 終了条件（すべての触媒が選ばれた場合）
        if len(self.remaining_indices) == 0:
            self.done = True

        return next_state, actual_reward, self.done

    def get_remaining_features(self):
        """ 残っている触媒の特徴量を返す """
        return self.features[self.remaining_indices]
    
    def get_remaining_indices(self):
        """ 残っている触媒のインデックスを返す """
        return self.remaining_indices

        
        
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
    def update_q_values(self, optimizer, criterion, state_tensor, actual_reward, predicted_reward, action, gamma, next_state_tensor):
        # 現在のQ値を計算
        q_values = self(state_tensor)
    
        # ターゲットの作成
        target = q_values.clone().detach()
        
        # predicted_reward(予測収率)に基づいて報酬を設定
        reward = predicted_reward
            
        # 報酬をテンソルに変換
        actual_reward_tensor = torch.tensor(reward, dtype=torch.float32)
 
        # 次の状態のQ値の計算（次の状態が与えられていれば）
        with torch.no_grad():
            if next_state_tensor is None or next_state_tensor.numel() == 0:  # 次の状態がない場合
                next_state_value = 0.0  # 次の状態がないので報酬はゼロ
            else:
                next_q_values = self(next_state_tensor)
                next_state_value = next_q_values.max().item()   # 最大Q値を取得
        

        # ターゲットQ値を計算（Q(s, a) = r + γ * max(Q(s', a'))）
        target[action] = actual_reward_tensor + gamma * next_state_value

        # 損失を計算
        loss = criterion(q_values, target)
 
        # ネットワークの更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()