#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#----------------------------------------------------------------------
# インポート
#----------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#----------------------------------------------------------------------
# クラス定義
#----------------------------------------------------------------------
class GraftRegressionTree:
    """
    接ぎ木のクラス
    """

    def __init__(self, tdtree):
        """
        コンストラクタ

        Args:
          tdtree：時間依存木の配列
        """

        # Public Member
        self.tdtree = tdtree
        self.max_depth = tdtree[0].tree_df.iloc[-1]['depth'] + 1
        self.tree_df = pd.DataFrame(columns=['tdtree_index',  'pred',  'depth', 'eval', 'feature_index', 'threshold', 'leaf'], index=range(self.__count_node(self.max_depth)[0]))

    #---------------------------------------
    # public関数
    #---------------------------------------

    def build_tree(self, tdtree_index_ls):
        """
        木を構築する
        
        Args:
          tdtree_index_ls：採用する時間依存木の種類
        """
        for time in range(len(tdtree_index_ls)):
            # 木の剪定（構造を取得）
            target_tdtree = self.tdtree[tdtree_index_ls[time]].tree_df
            target_tdtree_nodes = target_tdtree[target_tdtree['depth'] == time]
            # 構造をgrtreeに格納
            for i in self.__count_node(time+1)[1]:
                self.tree_df['tdtree_index'][i] = tdtree_index_ls[time]
                self.tree_df['pred'][i] = target_tdtree_nodes['mean'][i]
                self.tree_df['depth'][i] = target_tdtree_nodes['depth'][i]
                self.tree_df['eval'][i] = target_tdtree_nodes['eval'][i]
                self.tree_df['feature_index'][i] = target_tdtree_nodes['feature_index'][i]
                self.tree_df['threshold'][i] = target_tdtree_nodes['threshold'][i]
                self.tree_df['leaf'][i] = target_tdtree_nodes['leaf'][i]
        # 接ぎ木した後のNaNチェック
        for node_index in range(self.__count_node(self.max_depth-1)[0]):
            left_node_index = node_index*2 + 1
            right_node_index = node_index*2 + 2
            # もしleafがTrueなら問題なし
            if not self.tree_df['leaf'][node_index]:
                if not np.isnan(self.tree_df['pred'][node_index]):
                    if np.isnan(self.tree_df['pred'][left_node_index]) or np.isnan(self.tree_df['pred'][right_node_index]):
                        self.tree_df['leaf'][node_index] = True

    def predict(self, x_test_ls):
        """
        予測する
        
        """
        pred_array = []
        
        for pred_target_index in range(x_test_ls[0].shape[0]):
            current_node_index = 0
            for time in range(self.max_depth):
                target_tdtree_index = self.tree_df.iloc[current_node_index]['tdtree_index']
                # 葉であれば
                if self.tree_df.loc[current_node_index, 'leaf']:
                    pred_array.append(self.tree_df.loc[current_node_index, 'pred'])
                    break;
                # 節であれば
                elif x_test_ls[target_tdtree_index][pred_target_index][time] < self.tree_df.loc[current_node_index, 'threshold']:
                    current_node_index = current_node_index*2 + 1
                else:
                    current_node_index = current_node_index*2 + 2
                    
        return pred_array
    
    def evalution(self, y_true, y_pred, method):
        if method == "mse":
            return mean_squared_error(y_true, y_pred)
        
    #---------------------------------------
    # private methods
    #---------------------------------------
    
    def __count_node(self, max_depth):
        """
        深さに応じた総ノード数を計算する
        
        Args:
          max_depth：木の深さ

        Returns:
          総ノード数とその深さに該当するノードのインデックス

        """
        node_num = 1
        for i in range(max_depth):
            if i == 0:
                node_num
                depth_node_array = [0]
            else:
                pre_num = node_num
                node_num = node_num + 2**i
                depth_node_array = list(range(pre_num, node_num))

        return [node_num, depth_node_array]

