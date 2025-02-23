# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:59:16 2024

@author: noyor
"""

import pandas as pd

class DatasetSelect:
    """
    異なる反応間のドメイン適用に対応　任意でソースドメイン群を選択できます
    
    事前に必要なデータセットのｃｓｖファイルを呼び出しておく。
    'name' = pd.read_csv(ファイル名)
    nemeはワンホットエンコーダーで１になっている列名と一致させる。
    """
    def __init__(self):
        # 辞書としてデータセットを保存
        self.dataset_dict = {}
        # 相関の計算結果を保存
        self.correlation_matrix = pd.DataFrame()
        # 分割したデータを保持
        self.target_test = pd.DataFrame()
        self.target_train = pd.DataFrame()
   
        
    def import_dataset(self, df, name):
        """
        parameters:
            df : pd.DataFrame
            name : 'name of data'
        """
        self.dataset_dict[name] = df
        
    
    def calculate_corr(self, selected_catalyst_index):
        # 選択した触媒のみのデータセットを作成
        selected_catalyst_dataset_dict = {}
        
        for name, df in self.dataset_dict.items():
            selected_catalyst_dataset_dict[name] = df.loc[selected_catalyst_index]
            
        # Yieldの相関を計算
        target_column = 'Yield'
        
        col_data = {name: df[target_column] for name, df in selected_catalyst_dataset_dict.items()}
        col_df = pd.DataFrame(col_data)
        self.correlation_matrix = col_df.corr()
    
        
    def select_source_data(self, target_data_name, source_data_candidates, selection_order, k):
        """
        使用するソースデータを選択する。       
            parameters:
                target_data_name = "データ名"
                source_data_candidates = list of "ソースデータの候補"
                selection_order = "top" or "bottom"
                k = 選択するソースデータの数            
            returns:
                source_data_names = list of names for the source data
        """
        target_row = self.correlation_matrix.loc[target_data_name]
        target_row = target_row[target_row.index.isin(source_data_candidates)]
        
        if selection_order == "top":
            source_data_names = target_row.nlargest(k).index.tolist()
            
        elif selection_order == "bottom":
            source_data_names = target_row.nsmallest(k).index.tolist()
            
        else :
            print("This selection_order is not defined")
            
        return source_data_names
  
    def split_df(self, source_data_names, target_data_name):
        """
        データを分ける。

        """
        target_df = self.dataset_dict[target_data_name]
        target_df = target_df.drop(columns=['Name', 'ID'])
        
        selected_dfs = [df for key, df in self.dataset_dict.items() if any(name in key for name in source_data_names)]
        source_df = pd.concat(selected_dfs, ignore_index=True)
        source_df = source_df.drop(columns=['Name', 'ID'])
    
        ys = pd.DataFrame(source_df['Yield'],columns=['Yield'])
        Xs = source_df.drop(columns=['Yield'])

        yt = pd.DataFrame(target_df['Yield'],columns=['Yield'])
        Xt = target_df.drop(columns=['Yield'])
        
        return Xs, ys, Xt, yt
        
    
    def shaping_df(self, source_data_names, target_data_name):
        """
        使用しない反応の列を削除し、データを分ける。
            parameters
                source_data_names = list of names for the source data
                target_data_name = "データ名"     
            returns
                Xs = source data excluding reward (df)
                ys = reward of source data (df)
                Xt = target data ecxcluding reward (df)
                yt = reward of target data (df)
        """
        target_df = self.dataset_dict[target_data_name]
        target_df = target_df.drop(columns=['Name', 'ID'])
        
        selected_dfs = [df for key, df in self.dataset_dict.items() if any(name in key for name in source_data_names)]
        source_df = pd.concat(selected_dfs, ignore_index=True)
        source_df = source_df.drop(columns=['Name', 'ID'])
        
        column_names_to_keep = []
        column_names_to_keep.extend(source_data_names)
        column_names_to_keep.append(target_data_name)
        column_names_to_keep.append('Yield')
        
        target_df = target_df.loc[:, (target_df != 0).any(axis=0) | target_df.columns.isin(column_names_to_keep)]
        source_df = source_df.loc[:, (source_df != 0).any(axis=0) | source_df.columns.isin(column_names_to_keep)]
        
        ys = pd.DataFrame(source_df['Yield'],columns=['Yield'])
        Xs = source_df.drop(columns=['Yield'])

        yt = pd.DataFrame(target_df['Yield'],columns=['Yield'])
        Xt = target_df.drop(columns=['Yield'])
        
        return Xs, ys, Xt, yt
    

        