# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:08:15 2024

@author: noyor
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adapt.instance_based import TrAdaBoostR2
from sklearn.ensemble import RandomForestRegressor as rf


class OPSsearch_env:
    def __init__(self, features, rewards, epsilon):
        self.features = features  
        self.rewards = rewards   
        self.remaining_indices = list(range(len(features)))       
        self.done = False
        self.count = 0         
        
    def reset(self):
        self.remaining_indices = list(range(len(self.features)))
        self.done = False
        return self.remaining_indices
    
    def select_action(self, model, epsilon):
        """
        Determining the action based on the epsilon-greedy method
        epsilon : a catalyst is randomly selected
        1-epsilon: the catalyst with the highest predicted yield is selected.
        """
        random_value = random.random()
        print(f"Random value used for epsilon-greedy: {random_value:.2f}")   
        if random_value < epsilon:
            print("=======================================")
            print(f"random_value < {epsilon} => **random search**")
            print("=======================================")
                
            random_index = np.random.choice(len(self.remaining_indices))
            best_catalyst_remaining_index = random_index
            best_catalyst_actual_index = self.remaining_indices[random_index]
        else:
            predicted_rewards = model.predict(pd.DataFrame(self.get_remaining_features()))
            predicted_rewards_1d = predicted_rewards.ravel()
            best_catalyst_remaining_index = np.argmax(predicted_rewards_1d)
            best_catalyst_actual_index = self.remaining_indices[best_catalyst_remaining_index]
    
        self.count += 1.
        return best_catalyst_remaining_index, best_catalyst_actual_index
    
    def step(self, chosen_index):
        """
        Applies the chosen action and returns the next state, the actual reward, and the done flag.
        chosen_index: the actual index of the selected catalyst in the original dataset.
        """
        actual_reward = self.rewards[chosen_index]
        self.remaining_indices.remove(chosen_index)
        next_state = self.remaining_indices
        if len(self.remaining_indices) == 0:
            self.done = True

        return next_state, actual_reward, self.done
    
    def get_remaining_features(self):
        return self.features[self.remaining_indices]
    
    def get_remaining_indices(self):
        return self.remaining_indices
    
    def visualize_predicted_rewards(self, model):
        predicted_rewards = model.predict(pd.DataFrame(self.get_remaining_features()))
        predicted_rewards_1d = predicted_rewards.ravel()
        predicted_rewards_df = pd.DataFrame(columns=["Predicted_Yield", "ID"])
        for i, (predicted_reward, actual_index) in enumerate(zip(predicted_rewards_1d, self.remaining_indices)):
            ID_n = actual_index + 1
            predicted_rewards_df.loc[i] = [predicted_reward, f"OPS{ID_n}"]
        predicted_rewards_df.sort_values(by="Predicted_Yield", ascending=False, inplace=True)
        
        print("The top 30 OPSs with the highest predicted yields are then displayed.")
        print(predicted_rewards_df.head(30))
        print("\033[1mOne round of the search is completed...\033[0m")
        print("~*~*" * 20)


class DatasetSelect:
    def __init__(self):
        self.dataset_dict = {}
        self.correlation_matrix = pd.DataFrame()
        self.target_test = pd.DataFrame()
        self.target_train = pd.DataFrame()

    def import_dataset(self, df, name):
        self.dataset_dict[name] = df

    def calculate_corr(self, selected_catalyst_index):
        selected_catalyst_dataset_dict = {}
        for name, df in self.dataset_dict.items():
            selected_catalyst_dataset_dict[name] = df.loc[selected_catalyst_index]
        target_column = 'Yield'
        col_data = {name: df[target_column] for name, df in selected_catalyst_dataset_dict.items()}
        col_df = pd.DataFrame(col_data)
        self.correlation_matrix = col_df.corr()

    def select_source_data(self, target_data_name, source_data_candidates, selection_order, n_source):
        target_row = self.correlation_matrix.loc[target_data_name]
        target_row = target_row[target_row.index.isin(source_data_candidates)]
        if selection_order == "top":
            source_data_names = target_row.nlargest(n_source).index.tolist()
        elif selection_order == "bottom":
            source_data_names = target_row.nsmallest(n_source).index.tolist()
        else:
            print("This selection_order is not defined")
        return source_data_names

    def split_df(self, source_data_names):
        selected_dfs = [df for key, df in self.dataset_dict.items() if any(name in key for name in source_data_names)]
        source_df = pd.concat(selected_dfs, ignore_index=True)
        source_df = source_df.drop(columns=['Name', 'ID'])
        ys = pd.DataFrame(source_df['Yield'], columns=['Yield'])
        Xs = source_df.drop(columns=['Yield'])
        return Xs, ys
    
    def visualize_correlation(self, target):
        correlation_row = self.correlation_matrix[[target]].T
        plt.figure(figsize=(10, 1))
        sns.heatmap(correlation_row, annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
        plt.yticks(rotation=0)
        plt.title("Correlation Matrix of Yield Values")
        plt.show()
    

def TrABsearch(target, n_OPS=1, epsilon=0.1, es=rf(), ne=5, selection_order='top', n_source=5,
               data_dir="../../data/data_60", result_dir="../results/TrABsearch_top5_e10%_ne5"):
    print(f'Running TrABsearch for {target}...')

    All_data_names = ['Reaction_CO_1.5h', 'Reaction_CO_biphenyl', 'Reaction_CO_ortho',
                      'Reaction_CO_Cl', 'Reaction_CS', 'Reaction_CN', 'Reaction_2+2', 'Reaction_CF3',
                      'Reaction_CH2CF3', 'Reaction_CH2F', 'Reaction_Cy', 'Reaction_SCF3',
                      'Reaction_P']
    
    dfs = {name: pd.read_csv(f'{data_dir}/data_60_{name}.csv') for name in All_data_names}
    df_target = dfs[target]
    t_features = df_target.drop(columns=['Name', 'ID', 'Yield']).values
    t_rewards = df_target[['Yield']].values
    source_dfs = [dfs[name] for name in All_data_names if name != target]
    df_initial_source = pd.concat(source_dfs, ignore_index=True)
    
    ds = DatasetSelect()
    for data_name in All_data_names:
        ds.import_dataset(dfs[f'{data_name}'], data_name)

    s_initial_features = df_initial_source.drop(columns=['Name', 'ID', 'Yield']).values
    s_initial_rewards = df_initial_source[['Yield']].values

    All_source_data_names = [item for item in All_data_names if item != target]

    output_data = []
    for i in range(n_OPS):        
        initial_indices = [i]
        OPS_n = i + 1
        print(f"\033[1mStep 1: OPS{OPS_n} is selected.\033[0m")

        random.seed(42)
        np.random.seed(42)

        initial_t_features = t_features[initial_indices]
        initial_t_rewards = t_rewards[initial_indices]

        model = TrAdaBoostR2(es, n_estimators=ne, Xt=initial_t_features, yt=initial_t_rewards, random_state=42, verbose=0)
        model.fit(s_initial_features, s_initial_rewards)

        env = OPSsearch_env(features=t_features, rewards=t_rewards, epsilon=epsilon)

        for index in sorted(initial_indices, reverse=True):
            if index in env.remaining_indices:
                env.remaining_indices.remove(index)

        selected_t_indices = initial_indices.copy()
        selected_t_features = initial_t_features.tolist()
        selected_t_rewards = initial_t_rewards.tolist()
    
    
        actual_t_rewards_per_step = []
        predicted_t_rewards_per_step = []
        selected_t_indices_per_step = []
        for step in range(59):
            if step == 0:
                env.visualize_predicted_rewards(model)
                best_catalyst_remaining_index, best_catalyst_actual_index = env.select_action(model, epsilon)
                if best_catalyst_actual_index not in env.get_remaining_indices():
                    continue
            
                OPS_n = best_catalyst_actual_index + 1
                best_catalyst_feature = pd.DataFrame([t_features[best_catalyst_actual_index]])
                predicted_t_reward = model.predict(best_catalyst_feature).item()

                next_state, actual_t_reward, done = env.step(best_catalyst_actual_index)
                print(f"\033[1mStep {step+2}: OPS{OPS_n} is selected (Pred Yield: {predicted_t_reward:.0f}%, Actual Yield: {actual_t_reward.item()}%).\033[0m")

                actual_t_rewards_per_step.append(actual_t_reward.item())
                predicted_t_rewards_per_step.append(predicted_t_reward)
                selected_t_indices_per_step.append(best_catalyst_actual_index)

                selected_t_indices.append(best_catalyst_actual_index)
                selected_t_features.append(t_features[best_catalyst_actual_index].tolist())
                selected_t_rewards.append(actual_t_reward)
                
                selected_indices = initial_indices + selected_t_indices_per_step
                print("After obtaining the experimental yield, a TrAB model is constructed to predict the expected yields of the remaining OPSs.")
                print("First, correlation coefficients are calculated.")
                ds.calculate_corr(selected_catalyst_index=selected_indices)
                ds.visualize_correlation(target)
                print("====================================================================================")
                print("Since there are only two data points, the correlation coefficient is either 1 or -1.")
                print("Therefore, for Step 3, all reactions are used as the source domain.")
                print("====================================================================================")
                
                model = TrAdaBoostR2(es, n_estimators=ne, Xt=selected_t_features, yt=selected_t_rewards, random_state=42, verbose=0)
                model.fit(s_initial_features, s_initial_rewards)
                
                
            elif 1 <= step:
                env.visualize_predicted_rewards(model)
                best_catalyst_remaining_index, best_catalyst_actual_index = env.select_action(model, epsilon)
                if best_catalyst_actual_index not in env.get_remaining_indices():
                    continue
                
                OPS_n = best_catalyst_actual_index + 1
                best_catalyst_feature = pd.DataFrame([t_features[best_catalyst_actual_index]])
                predicted_t_reward = model.predict(best_catalyst_feature).item()

                next_state, actual_t_reward, done = env.step(best_catalyst_actual_index)
                print(f"\033[1mStep {step+2}: OPS{OPS_n} is selected (Pred Yield: {predicted_t_reward:.0f}%, Actual Yield: {actual_t_reward.item()}%).\033[0m")

                actual_t_rewards_per_step.append(actual_t_reward)
                predicted_t_rewards_per_step.append(predicted_t_reward)
                selected_t_indices_per_step.append(best_catalyst_actual_index)

                selected_t_indices.append(best_catalyst_actual_index)
                selected_t_features.append(t_features[best_catalyst_actual_index].tolist())
                selected_t_rewards.append(actual_t_reward)
                
                selected_indices = initial_indices + selected_t_indices_per_step
                print("After obtaining the experimental yield, a TrAB model is constructed to predict the expected yields of the remaining OPSs.")
                print("For selecting the source domain, correlation coefficients are calculated.")
                ds.calculate_corr(selected_catalyst_index=selected_indices)
                ds.visualize_correlation(target)
                
                source_data_names = ds.select_source_data(target_data_name = target,
                                                                 source_data_candidates = All_source_data_names,
                                                                 selection_order = selection_order,
                                                                 n_source = n_source)
                Xs, ys = ds.split_df(source_data_names = source_data_names)
                print("==============================================================================================")
                print("Selected reactions as the source domain:")
                print(f"{source_data_names}")
                print("==============================================================================================")

                model = TrAdaBoostR2(es, n_estimators=ne, Xt=selected_t_features, yt=selected_t_rewards, random_state=42, verbose=0)
                model.fit(Xs, ys)
                
                if step == 4:
                    print("\033[1m  ·\033[0m")
                    print("\033[1m  ·\033[0m")
                    print("\033[1mRepeat until all OPSs are selected.\033[0m")
                    break
                       
        output_data.append({'initial_indices': initial_indices,
                            'initial_reward': initial_t_rewards.item(),
                            'selected_indices_per_step': selected_t_indices_per_step,
                            'actual_rewards_per_step': actual_t_rewards_per_step,
                            'predicted_rewards_per_step': predicted_t_rewards_per_step})
        
        if i == 0:
            print("\033[1m  ·\033[0m")
            print("\033[1m  ·\033[0m")
            print("\033[1m  ·\033[0m")
            print("\033[1m  ·\033[0m")
            print("\033[1mRepeat until each OPS has been used once as the initially selected OPS.\033[0m")
            break
        
    df_result = pd.DataFrame(output_data)
    df_out = pd.DataFrame()
    
    df_out['selected_indices_per_step'] = df_result['initial_indices'] + df_result['selected_indices_per_step']
    df_out['actual_rewards_per_step'] = df_result.apply(lambda row: [row['initial_reward']] + list(row['actual_rewards_per_step']), axis=1)
    df_out['predicted_rewards_per_step'] = df_result['predicted_rewards_per_step']
    
    out_file = os.path.join(result_dir, f"TrABsearch_{target}_{selection_order}{n_source}_e{epsilon}_ne{ne}_result.csv")
    df_out.to_csv(out_file, index=False)
