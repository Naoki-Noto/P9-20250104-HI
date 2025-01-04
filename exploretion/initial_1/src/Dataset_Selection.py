# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:59:16 2024

@author: noyor
"""

import pandas as pd
from sklearn.model_selection import train_test_split

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
        
    def split_target_data(self, target_data_name, test_size, random_state):
        """
        Parameters
        ----------
        target_data_name : 'name of data'
        test_size : number of testsize (0 ~ 1)
        random_state : random_state

        Returns
        -------
        target_train : target_train 
        
        target_test : target_test
        """
        target_df = self.dataset_dict[target_data_name]
        self.target_train, self.target_test = train_test_split(target_df, test_size=test_size, random_state=random_state)
        return self.target_train, self.target_test
    
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
    
    def calculate_corr_test_train(self):
        # 選択した触媒のみのデータセットを作成
        selected_catalyst_dataset_dict = {}
        
        # `id` 列の値を取得
        selected_ids = self.target_train['ID'].unique()
        
        # フィルタリング処理
        for name, df in self.dataset_dict.items():
            selected_catalyst_dataset_dict[name] = df[df['ID'].isin(selected_ids)]
            
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
    
    def shaping_df_test_train(self, source_data_names, target_data_name):
        """
        使用しない反応の列を削除し、データを分ける。
            parameters
                source_data_names = list of names for the source data
                target_data_name = "データ名"     
            returns
                Xs = source data excluding reward (df)
                ys = reward of source data (df)
                Xt_train = target train data ecxcluding reward (df)
                yt_train = reward of target train data (df)
                Xt_test = target test data ecxcluding reward (df)
                yt_train = reward of target test data (df)
        """

        target_train = self.target_train.drop(columns=['Name', 'ID'])
        target_test = self.target_test.drop(columns=['Name', 'ID'])
        
        selected_dfs = [df for key, df in self.dataset_dict.items() if any(name in key for name in source_data_names)]
        source_df = pd.concat(selected_dfs, ignore_index=True)
        source_df = source_df.drop(columns=['Name', 'ID'])
        
        column_names_to_keep = []
        column_names_to_keep.extend(source_data_names)
        column_names_to_keep.append(target_data_name)
        column_names_to_keep.append('Yield')
        
        target_train = target_train.loc[:, (target_train != 0).any(axis=0) | target_train.columns.isin(column_names_to_keep)]
        target_test = target_test.loc[:, (target_test != 0).any(axis=0) | target_test.columns.isin(column_names_to_keep)]
        source_df = source_df.loc[:, (source_df != 0).any(axis=0) | source_df.columns.isin(column_names_to_keep)]
        
        ys = pd.DataFrame(source_df['Yield'],columns=['Yield'])
        Xs = source_df.drop(columns=['Yield'])

        yt_train = pd.DataFrame(target_train['Yield'],columns=['Yield'])
        Xt_train = target_train.drop(columns=['Yield'])
        
        yt_test = pd.DataFrame(target_test['Yield'],columns=['Yield'])
        Xt_test = target_test.drop(columns=['Yield'])
        
        return Xs, ys, Xt_train, yt_train, Xt_test, yt_test
        