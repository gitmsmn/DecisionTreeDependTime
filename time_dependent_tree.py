#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#----------------------------------------------------------------------
# インポート
#----------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#----------------------------------------------------------------------
# クラス定義
#----------------------------------------------------------------------
class TimeDependentTree:
    """
    時間依存木のクラス
    """

    def __init__(self, x_train, y_train, x_test, y_test, max_depth, min_samples_leaf):
        """
        コンストラクタ

        Args:
          x_train：訓練データの説明変数
          y_train：訓練データの目的変数
          x_test：検証データの説明変数
          y_test：検証データの目的変数
          max_depth：木の最大の深さ
        """

        # Public Member
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_df = pd.DataFrame(columns=['n',  'mean',  'depth', 'eval', 'feature_index', 'threshold', 'leaf'], 
                       index=range(self.__count_node(max_depth)[0]))
        self.tree_ls = [[np.array([0]) for _ in range(2)] for _ in range(self.__count_node(self.max_depth)[0])]

    #---------------------------------------
    # クラス定数
    #---------------------------------------


    #---------------------------------------
    # public関数
    #---------------------------------------

    def train(self):
        """
        時間依存木の訓練
        
        """
        # 前処理
        self.tree_ls[0][0] = self.x_train
        self.tree_ls[0][1] = self.y_train
        self.tree_df.loc[:, 'leaf'] = False
        for time in range(self.max_depth):
            for node_index in self.__count_node(time+1)[1]:
                self.tree_df.loc[node_index, 'depth'] = time
        
        # 学習
        for time in range(self.max_depth):
            for node_index in self.__count_node(time+1)[1]:

                left_node_index = node_index*2 + 1
                right_node_index = node_index*2 + 2

                # 存在するノードなら
                if self.tree_df.loc[node_index, 'n'] != -1:
                    # 最下層でなければ分割点を探す
                    if self.tree_df.loc[node_index, 'depth'] < self.max_depth-1:
                        split_point_array = self.__best_split_fixed_depth(ls=self.tree_ls, 
                                                                           node_index=node_index,
                                                                           eval_type="MSE",
                                                                           time=time, 
                                                                           min_samples_leaf=self.min_samples_leaf)
                    # 最下層であれば葉として記録する準備をする
                    else:
                        split_point_array = [np.nan, np.nan, np.nan, True]

                    # ノードの情報を記録する
                    self.tree_df = self.__record_df(df=self.tree_df, 
                                                    ls=self.tree_ls, 
                                                    node_index=node_index, 
                                                    time=time, 
                                                    split_array=split_point_array)

                    # 直前の実行内容で葉に切り替わったら
                    if self.tree_df.loc[node_index, 'leaf']:
                        # その先のノードは存在しない扱いにする
                        self.tree_df.loc[left_node_index, 'n'] = -1
                        self.tree_df.loc[right_node_index, 'n'] = -1
                    # まだ節 かつ 最下層でなければ
                    elif self.tree_df.loc[node_index, 'leaf'] == False and self.tree_df.loc[node_index, 'depth'] < self.max_depth-1:
                        # その先のノードにデータを付与する
                        self.tree_ls = self.__record_next_ls(df=self.tree_df, 
                                                            ls=self.tree_ls, 
                                                            node_index=node_index, 
                                                            time=time,
                                                            split_array=split_point_array)
       
                # 存在しないノードならば
                else:
                    self.tree_df.loc[left_node_index, 'n'] = -1
                    self.tree_df.loc[right_node_index, 'n'] = -1
                
        # 後処理
        self.tree_df = self.tree_df[:self.__count_node(self.max_depth)[0]]
        
    def predict(self):
        """
        時間依存木の予測
        
        """
        pred_array = []
        
        for pred_target_index in range(self.x_test.shape[0]):            
            current_depth = 0
            for i in range(self.max_depth):
                # 葉であれば
                if self.tree_df.loc[current_depth, 'leaf']:
                    pred_array.append(self.tree_df.loc[current_depth, 'mean'])
                    break;
                # 節であれば
                elif self.x_test[pred_target_index][i] < self.tree_df.loc[current_depth, 'threshold']:
                    current_depth = current_depth*2 + 1
                else:
                    current_depth = current_depth*2 + 2
                    
        return pred_array

    #---------------------------------------
    # private methods
    #---------------------------------------
    
    def __best_split_fixed_depth(self, ls, node_index, eval_type, time, min_samples_leaf):
    
        best_evaluation = 10**8
        best_feature_index = -1
        best_threshold = None
        is_leaf = False
        num_df_row = ls[node_index][0].shape[0]

        thresholds, values = zip(*sorted(zip(ls[node_index][0][:, time], ls[node_index][1])))

        # 予測対象数だけループ
        for i in range(1, num_df_row):
            tentative_thresholds = thresholds[i - 1]
            left_node = values[0:i]
            right_node = values[i:]
            left_pred = np.full(len(left_node), np.mean(left_node))
            right_pred = np.full(len(right_node), np.mean(right_node))

            if(eval_type == "MSE"):
                evaluation = self.__mean_squared_error(left_pred, left_node) + self.__mean_squared_error(right_pred, right_node)

                if best_evaluation > evaluation and left_pred.shape[0] > min_samples_leaf and right_pred.shape[0] > min_samples_leaf:
                    best_evaluation = evaluation
                    best_feature_index = time
                    best_threshold = tentative_thresholds

            elif(eval_type == "KLD_sum"):
                evaluation = calc_kld(left_pred, left_node) + calc_kld(right_pred, right_node)

                if best_evaluation > evaluation and left_pred.shape[0] > min_samples_leaf and right_pred.shape[0] > min_samples_leaf:
                    best_evaluation = evaluation
                    best_feature_index = time
                    best_threshold = tentative_thresholds

            elif(eval_type == "KLD_def"):
                evaluation = abs(calc_kld(left_pred, left_node) - calc_kld(right_pred, right_node))

                if best_evaluation > evaluation and left_pred.shape[0] > min_samples_leaf and right_pred.shape[0] > min_samples_leaf:
                    best_evaluation = evaluation
                    best_feature_index = time
                    best_threshold = tentative_thresholds

            else:
                break

        if best_evaluation == 10**8:
            is_leaf = True

        return [best_evaluation, best_feature_index, best_threshold, is_leaf]

    
    def __record_df(self, df, ls, node_index, time, split_array):
        df.loc[node_index, 'n'] = len(ls[node_index][1])
        df.loc[node_index, 'mean'] = np.mean(ls[node_index][1])
        df.loc[node_index, 'depth'] = time
        df.loc[node_index, 'eval'] = split_array[0]
        df.loc[node_index, 'feature_index'] = split_array[1]
        df.loc[node_index, 'threshold'] = split_array[2]
        df.loc[node_index, 'leaf'] = split_array[3]

        return df

    
    def __record_next_ls(self, df, ls, node_index, time, split_array):
    
        LEFT_NODE_INDEX = node_index*2 + 1
        RIGHT_NODE_INDEX = node_index*2 + 2

        concat_xy = np.hstack([ls[node_index][0], ls[node_index][1].reshape(len(ls[node_index][1]), 1)])
        left_node = concat_xy[concat_xy[:, df.loc[node_index, 'feature_index']] < df.loc[node_index, 'threshold']]
        right_node = concat_xy[concat_xy[:, df.loc[node_index, 'feature_index']] >= df.loc[node_index, 'threshold']]

        ls[LEFT_NODE_INDEX][0] = left_node[:, :-1]
        ls[LEFT_NODE_INDEX][1] = left_node[:, -1]
        ls[RIGHT_NODE_INDEX][0] = right_node[:, :-1]
        ls[RIGHT_NODE_INDEX][1] = right_node[:, -1]

        return ls

    
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
    
    def __mean_squared_error(self, y_true, y_pred):
        """
        MSEを計算する

        """
        return np.mean((y_true - y_pred) ** 2)

