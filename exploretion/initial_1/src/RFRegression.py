import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random
import pandas as pd

All_data_names = ['Reaction_CO_1.5h', 'Reaction_CO_7.5h', 'Reaction_CO_biphenyl', 'Reaction_CO_ortho', 'Reaction_CO_Cl', 'Reaction_CS', 
                  'Reaction_CN', 'Reaction_2+2', 'Reaction_CF3', 'Reaction_CH2CF3', 'Reaction_CH2F', 'Reaction_Cy', 'Reaction_OCH2F', 'Reaction_SCF3',
                 'Reaction_Masuda', 'Reaction_Sumida']

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
        
        initial_indices = random.sample(range(60), 1)
        initial_features = features[initial_indices]
        initial_rewards = rewards[initial_indices]
        
        # 残りの触媒
        remaining_indices = list(set(range(60)) - set(initial_indices))
        remaining_features = features[remaining_indices]
        
        # ランダムフォレストモデルの初期化
        model = RandomForestRegressor(random_state=42)
        # 初期の5つの触媒データで線形回帰モデルを訓練
        model.fit(initial_features, initial_rewards)
        
        # 選択した触媒を保存するリスト（最初は5つの初期データ）
        selected_indices = initial_indices.copy()
        
        # タイムステップごとの実際の収率と予測収率を保存するリスト
        selected_indices_per_step = []
        actual_rewards_per_step = []
        predicted_rewards_per_step = []
        
        # 操作を繰り返すループ
        for step in range(59):  # 最終的に100個全ての触媒を使うまで繰り返す
            # モデルを使って残りの触媒に対する予測を行う
            predicted_rewards = model.predict(remaining_features)
            
            # 最も高い予測収率を持つ触媒を選択
            best_catalyst_index_in_remaining = np.argmax(predicted_rewards)
            
            # 予測収率の高い触媒を選び、予測値を報酬として取得
            best_catalyst_feature = remaining_features[best_catalyst_index_in_remaining]
            best_catalyst_actual_index = remaining_indices[best_catalyst_index_in_remaining]  # 元のデータに対するインデックス
            predicted_reward = float(predicted_rewards[best_catalyst_index_in_remaining])  # スカラー値に変換
            actual_reward = float(rewards[best_catalyst_actual_index].item())  # スカラー値に変換
            
            # 新しく選んだ触媒を訓練データに追加
            initial_features = np.append(initial_features, [best_catalyst_feature], axis=0)
            initial_rewards = np.append(initial_rewards, rewards[best_catalyst_actual_index])
            
            # タイムステップごとの実際の収率と予測収率を記録
            actual_rewards_per_step.append(actual_reward)
            predicted_rewards_per_step.append(predicted_reward)
            selected_indices_per_step.append(best_catalyst_actual_index)
            
            # 選ばれた触媒のインデックスを追加して、次のステップで除外する
            selected_indices.append(best_catalyst_actual_index)
            remaining_indices.remove(best_catalyst_actual_index)
            remaining_features = features[remaining_indices]  # 残りの触媒を更新
            
            # モデルを再訓練（選ばれた触媒を追加した新しいモデルを作成）
            model.fit(initial_features, initial_rewards)
            
        
        output_data.append({'random_state': random_state,
                            'initial_indices': initial_indices, 
                            'selected_indices_per_step': selected_indices_per_step,
                            'actual_rewards_per_step':actual_rewards_per_step,
                            'predicted_rewards_per_step':predicted_rewards_per_step
                            })
        
    df = pd.DataFrame(output_data)
    df.to_csv(f'../results/RFRegression_{target}_result.csv', index=False)
    
    
