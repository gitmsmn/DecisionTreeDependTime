#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def mean_squared_error(y_true, y_pred):
    """
    MSEを計算する

    """
    return np.mean((y_true - y_pred) ** 2)


# In[ ]:


def calc_kld(generated_data, ground_truth, bins=20):
    """
    KLDを計算する

    """
    # 最小値の設定
    generated_data = np.array(generated_data)
    ground_truth = np.array(ground_truth)
    if ground_truth.min() < generated_data.min():
        range_min = ground_truth.min()
    else:
        range_min = generated_data.min()
    
    # 最大値の設定
    if ground_truth.max() < generated_data.max():
        range_max = generated_data.max()
    else:
        range_max = ground_truth.max()
    
    pd_gt, _ = np.histogram(ground_truth, bins=bins, density=True, range=(range_min, range_max))
    pd_gen, _ = np.histogram(generated_data, bins=bins, density=True, range=(range_min, range_max))
    kld = 0
    for x1, x2 in zip(pd_gt, pd_gen):
        if x1 != 0 and x2 == 0:
            kld += x1
        elif x1 == 0 and x2 != 0:
            kld += x2
        elif x1 != 0 and x2 != 0:
            kld += x1 * np.log(x1 / x2)

    return np.abs(kld)


# In[ ]:


def search_best_split(X, y, eval_type):
    """
    最適な分割点を探す。

    Parameters
    ----------
    X : np.array
        説明変数の配列。多変量を想定。
    y : np.array
        目的変数の配列。
    eval_type : string
        分割基準の指標。
    """
    _eveluations = []
    best_evaluation = 10**8
    num_df_column = X.shape[1]
    num_df_row = X.shape[0]

    #特徴量の数だけループ
    for feature_index in range(num_df_column):
        thresholds, values = zip(*sorted(zip(X[:, feature_index], y)))

        # 予測対象数だけループ
        for i in range(1, num_df_row):
            tentative_thresholds = thresholds[i - 1]
            left_node = values[0:i]
            right_node = values[i:]
            left_pred = np.full(len(left_node), np.mean(left_node))
            right_pred = np.full(len(right_node), np.mean(right_node))

            if(eval_type == "MSE"):
                evaluation = mean_squared_error(left_pred, left_node) + mean_squared_error(right_pred, right_node)
                _eveluations.append(evaluation)

                if best_evaluation > evaluation:
                    best_evaluation = evaluation
                    best_feature_index = feature_index
                    best_threshold = tentative_thresholds

            elif(eval_type == "KLD_sum"):
                evaluation = calc_kld(left_pred, left_node) + calc_kld(right_pred, right_node)
                _eveluations.append(evaluation)

                if best_evaluation > evaluation:
                    best_evaluation = evaluation
                    best_feature_index = feature_index
                    best_threshold = tentative_thresholds

            elif(eval_type == "KLD_def"):
                evaluation = abs(calc_kld(left_pred, left_node) - calc_kld(right_pred, right_node))
                _eveluations.append(evaluation)

                if best_evaluation > evaluation:
                    best_evaluation = evaluation
                    best_feature_index = feature_index
                    best_threshold = tentative_thresholds

            else:
                break
        
    return [best_evaluation, best_feature_index, best_threshold]


# In[ ]:


def make_wave(A, f, sec, sf):
    """
    sin波を作成する。

    Parameters
    ----------
    A : float
        振幅
    f : float
        周波数
    sec : float
        信号の長さ
    sf : float
        信号の長さ
    """
    
    wave_t = np.arange(0, sec, 1/sf) #サンプリング点の生成
    wave_y = A*np.sin(2*np.pi*f*wave_t) #正弦波の生成
    
    return wave_t, wave_y


# In[ ]:


def sliding_window(target_raw, window_size=5):
    """
    スライディングウィンドウ。

    """
    
    X, Y = [], []
    for i in range(len(target_raw)-window_size):
        X.append(target_raw[i:i + window_size])
        Y.append(target_raw[i + window_size])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


# In[ ]:


def best_split_fixed_depth(X, y, eval_type, feature_index, min_samples_leaf):
    """
    深さ固定で最適な分割点を探す。

    Parameters
    ----------
    X : np.array
        説明変数の配列。多変量を想定。
    y : np.array
        目的変数の配列。
    eval_type : string
        分割基準の指標。
    """
    best_evaluation = 10**8
    num_df_row = X.shape[0]

    thresholds, values = zip(*sorted(zip(X[:, feature_index], y)))

    # 予測対象数だけループ
    for i in range(1, num_df_row):
        tentative_thresholds = thresholds[i - 1]
        left_node = values[0:i]
        right_node = values[i:]
        left_pred = np.full(len(left_node), np.mean(left_node))
        right_pred = np.full(len(right_node), np.mean(right_node))

        if(eval_type == "MSE"):
            evaluation = mean_squared_error(left_pred, left_node) + mean_squared_error(right_pred, right_node)

            if best_evaluation > evaluation and left_pred.shape[0] > min_samples_leaf and right_pred.shape[0] > min_samples_leaf:
                best_evaluation = evaluation
                best_feature_index = feature_index
                best_threshold = tentative_thresholds

        elif(eval_type == "KLD_sum"):
            evaluation = calc_kld(left_pred, left_node) + calc_kld(right_pred, right_node)

            if best_evaluation > evaluation and left_pred.shape[0] > min_samples_leaf and right_pred.shape[0] > min_samples_leaf:
                best_evaluation = evaluation
                best_feature_index = feature_index
                best_threshold = tentative_thresholds

        elif(eval_type == "KLD_def"):
            evaluation = abs(calc_kld(left_pred, left_node) - calc_kld(right_pred, right_node))

            if best_evaluation > evaluation and left_pred.shape[0] > min_samples_leaf and right_pred.shape[0] > min_samples_leaf:
                best_evaluation = evaluation
                best_feature_index = feature_index
                best_threshold = tentative_thresholds

        else:
            break
    
    return [best_evaluation, best_feature_index, best_threshold]


# In[ ]:


# ノードのカウント
def count_node(max_depth):
    """
    深さに応じた総ノード数を計算する

    """
    node_num = 1
    depth_node_array = []
    
    for i in range(max_depth):
        if i == 0:
            depth_node_array = [0]
        else:
            pre_num = node_num
            node_num = node_num + 2**i
            depth_node_array = list(range(pre_num, node_num))
    
    return [node_num, depth_node_array]


# In[ ]:


# スライディングウィンドウ
def sliding_window(target_raw):
    X, Y = [], []
    window_size = 5
    for i in range(len(target_raw)-window_size):
        X.append(target_raw[i:i + window_size])
        Y.append(target_raw[i + window_size])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

