# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:48:42 2024

@author: noyor
"""

import pandas as pd
from edbo.utils import Data
from edbo.bro import BO_express
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


for target in ['Reaction_CO_1.5h', 'Reaction_CO_biphenyl', 'Reaction_CO_ortho', 'Reaction_CO_Cl', 
               'Reaction_CS', 'Reaction_CN', 'Reaction_2+2', 
               'Reaction_CF3', 'Reaction_CH2CF3', 'Reaction_CH2F', 'Reaction_Cy', 'Reaction_SCF3',
               'Reaction_OCH2F', 'Reaction_P', 'Reaction_Si']:
    
    print(f'Running BOsearch for {target}...')
    
    random.seed(42)

    file_path = f"../../data/data_BO_sc/data_60_{target}.csv"
    df = pd.read_csv(file_path)
    rewards = df['yield']
    features = df.drop(columns=['Name', 'ID', 'yield'])
    OPS_desc = Data(features)
    components = {'OPS': '<defined in desc>'}
    desc = {'OPS': OPS_desc.data}

    output_list = []
    for first_index in range(60):
        remaining_indices = set(range(len(OPS_desc.data)))
        selected_indices = [first_index]
        remaining_indices.remove(first_index)
        random.seed(first_index)
        second_index = random.choice(list(remaining_indices))
        remaining_indices.remove(second_index)
        selected_data = pd.DataFrame({'yield': [rewards.iloc[first_index], rewards.iloc[second_index]]}).reset_index(drop=True)
        selected_features = desc['OPS'].iloc[[first_index, second_index]].reset_index(drop=True)
        fs_selected_samples = pd.concat([selected_features, selected_data], axis=1)
        fs_selected_samples.insert(0, 'Index', [first_index, second_index])
        fs_selected_samples.drop(columns=["HOMO", "LUMO", "E_S1", "f_S1", "E_T1", "dEST", "dDM"], errors='ignore', inplace=True)
        fs_selected_samples.to_csv('temp_file/temp_initial.csv', index=False)
        bo = BO_express(reaction_components=components,
                        descriptor_matrices=desc,
                        acquisition_function='UCB',
                        init_method='rand',
                        batch_size=1,
                        target='yield')
        bo.add_results('temp_file/temp_initial.csv')
    
        fs_results = fs_selected_samples.drop(columns=["HOMO", "LUMO", "E_S1", "f_S1", "E_T1", "dEST", "dDM"], errors="ignore")

        index_list = []
        reward_list = []
        for i in range(58):
            bo.run()
            #bo.plot_convergence()
            #bo.model.regression()
            bo.export_proposed(f'temp_file/selected_OPS{i}.csv')
            selected_OPS = pd.read_csv(f'temp_file/selected_OPS{i}.csv')
            smiles_index = selected_OPS.loc[0, 'smiles_index']
            bo_index = df.query("smiles == @smiles_index").index[0]
            index_list.append(bo_index)
            reward_list.append(rewards.iloc[bo_index])
            selected_OPS.at[0, 'yield'] = rewards.iloc[bo_index]
            selected_OPS.to_csv(f'temp_file/temp{i}.csv', index=False)
            bo.add_results(f'temp_file/temp{i}.csv')

        indices = pd.DataFrame(index_list, columns=['Index'])
        final_rewards = pd.DataFrame(reward_list, columns=['yield'])
        result = pd.concat([indices, final_rewards], axis=1)
        final_result = pd.concat([fs_results, result], ignore_index=True)
        index_list = final_result["Index"].tolist()
        yield_list = final_result["yield"].tolist()
        output = pd.DataFrame({"selected_indices_per_step": [index_list],
                               "actual_rewards_per_step": [yield_list]})
        output_list.append(output)

    final_output = pd.concat(output_list, ignore_index=True)
    final_output.to_csv(f'../results/BOsearch_UCB_sc/BOsearch_{target}_result.csv', index=False)
    #print(f'{target}', final_output)
