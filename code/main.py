import collections
import gc
import math
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from operator import itemgetter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings('ignore')

data_path = '../data/' # 天池平台路径
save_path = '../output/'  # 天池平台路径

# # 全量训练集
train_click = pd.read_csv(data_path + 'train_click_log.csv')
testA_click = pd.read_csv(data_path + 'testA_click_log.csv')
train_click = train_click.append(testA_click)

test_click = pd.read_csv(data_path + 'testB_click_log.csv')
articles = pd.read_csv(data_path + 'articles.csv')

def get_all_click_df(data_path='./data_raw/', train=True, test=True):
    if train:
        all_click = train_click.copy()
    if test:
        all_click = test_click.copy()
    if train and test:
        trn_click = train_click.copy()
        tst_click = test_click.copy()
        all_click = trn_click.append(tst_click)
    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

#物品相似度召回
def itemcf_recall(topk=10):
    ts = time.time()
    def get_past_click():
        train = train_click.sort_values(['user_id', 'click_timestamp']).reset_index().copy()
        list1 = []
        train_indexs = []
        for user_id in tqdm(train['user_id'].unique()):
            user = train[train['user_id'] == user_id]
            row = user.tail(1)
            train_indexs.append(row.index.values[0])
            #testA中有一些只点了一次的用户要去掉
            if len(user) >= 2:
                list1.append(row.values.tolist()[0])
        train_last_click = pd.DataFrame(list1, columns=['index', 'user_id', 'article_id', 'click_timestamp', 'click_environment',\
                                        'click_deviceGroup', 'click_os', 'click_country', 'click_region',
                                        'click_referrer_type'])
        train_last_click = train_last_click.drop(columns=['index'])
        train_past_clicks = train[~train.index.isin(train_indexs)]
        train_past_clicks = train_past_clicks.drop(columns=['index'])
        
        test = test_click.sort_values(['user_id', 'click_timestamp']).reset_index().copy()
        list2 = []
        for user_id in tqdm(test['user_id'].unique()):
            user = test[test['user_id'] == user_id]
            row = user.tail(1)
            list2.append(row.values.tolist()[0])
        test_last_click = pd.DataFrame(list2, columns=['index', 'user_id', 'article_id', 'click_timestamp', 'click_environment',\
                                        'click_deviceGroup', 'click_os', 'click_country', 'click_region',
                                        'click_referrer_type'])
        test_last_click = test_last_click.drop(columns=['index'])
        
        ###                    注释要去掉↓
        all_click_df = train_past_clicks.append(test_click)
        all_click_df = all_click_df.reset_index().drop(columns=['index'])

        all_click_df = all_click_df.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
        return all_click_df, train_past_clicks, train_last_click, test_last_click

    # 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
    def get_user_item_time(click_df):
        
        click_df = click_df.sort_values('click_timestamp')
        
        def make_item_time_pair(df):
            return list(zip(df['click_article_id'], df['click_timestamp']))
        
        user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(lambda x: make_item_time_pair(x))\
                                                                .reset_index().rename(columns={0: 'item_time_list'})
        user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
        
        return user_item_time_dict

    def itemcf_sim(df):
        user_item_time_dict = get_user_item_time(df)
        
        # 计算物品相似度
        i2i_sim = {}
        item_cnt = defaultdict(int)
        for user, item_time_list in tqdm(user_item_time_dict.items()):
            # 在基于商品的协同过滤优化的时候可以考虑时间因素
            for i, i_click_time in item_time_list:
                item_cnt[i] += 1
                i2i_sim.setdefault(i, {})
                for j, j_click_time in item_time_list:
                    if(i == j):
                        continue
                    i2i_sim[i].setdefault(j, 0)
                    
                    i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)
                    
        i2i_sim_ = i2i_sim.copy()
        for i, related_items in i2i_sim.items():
            for j, wij in related_items.items():
                i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
        
        # 将得到的相似性矩阵保存到本地
        pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
        
        return i2i_sim_
    all_click_df, train_past_clicks, train_last_click, test_last_click = get_past_click()
    i2i_sim = itemcf_sim(all_click_df)

    # 基于商品的召回i2i
    def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num):
        # 获取用户历史交互的文章
        user_hist_items = user_item_time_dict[user_id]
        user_hist_items_ = {user_id for user_id, _ in user_hist_items}
        
        item_rank = {}
        for loc, (i, click_time) in enumerate(user_hist_items):
            for j, wij in i2i_sim[i][:sim_item_topk]:
                if j in user_hist_items_:
                    continue         
                item_rank.setdefault(j, 0)
                item_rank[j] +=  wij

        item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
            
        return item_rank

    # 定义
    user_recall_items_dict = collections.defaultdict(dict)

    # 获取 用户 - 文章 - 点击时间的字典
    user_item_time_dict = get_user_item_time(all_click_df)

    # 去取文章相似度
    i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))

    for i in tqdm(i2i_sim.keys()):
        i2i_sim[i] = sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)

    # 相似文章的数量
    sim_item_topk = topk

    # 召回文章数量
    recall_item_num = topk

    for user in tqdm(all_click_df['user_id'].unique()):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, 
                                                            sim_item_topk, recall_item_num)
    # 将字典的形式转换成df
    user_item_score_list = []

    for user, items in tqdm(user_recall_items_dict.items()):
        for item, score in items:
            user_item_score_list.append([user, item, score])

    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])
    recall_df.to_csv(save_path + 'recall_df.csv', index=False)

    # 从所有的召回数据中将测试集中的用户选出来
    tst_recall = recall_df[recall_df['user_id'].isin(test_last_click['user_id'].unique())]
    train_recall = recall_df[recall_df['user_id'].isin(train_last_click['user_id'].unique())]

    test_recall = tst_recall.copy()
    test_recall = test_recall.sort_values(by=['user_id', 'pred_score'])

    test_recall = test_recall.drop(columns=['pred_score'])

    test_recall.to_csv(save_path + 'itemcf_test_recall.csv', index=False)
    train_recall.to_csv(save_path + 'itemcf_train_recall.csv', index=False)

    print('Itemcf Recall Finished! Cost time: {}'.format(time.time() - ts))
    return train_past_clicks, train_last_click, test_last_click

#热度召回    
def hot_recall(topk=10, train_past_clicks=None, test_last_click=None):
    ts = time.time()
    
    train_click_df = get_all_click_df(data_path, test=False)
    test_click_df = get_all_click_df(data_path, train=False)
    
    train_click_df = train_click_df.sort_values(['user_id', 'click_timestamp'])
    test_click_df = test_click_df.sort_values(['user_id', 'click_timestamp'])
    
    articles_copy = articles.rename(columns={'article_id': 'click_article_id'})
    
    train_click_df = train_click_df.merge(articles_copy, on='click_article_id', how='left')
    test_click_df = test_click_df.merge(articles_copy, on='click_article_id', how='left')

    train_last_click = train_past_clicks.groupby('user_id').agg({'click_timestamp': 'max'}).reset_index()

    train_last_click_time = train_last_click.set_index('user_id')['click_timestamp'].to_dict()
    test_last_click_time = test_last_click.set_index('user_id')['click_timestamp'].to_dict()

    def get_item_topk_click_(hot_articles, hot_articles_dict, click_time, past_click_articles, k):
        topk_click = []
        min_time = click_time - 24 * 60 * 60 * 1000
        max_time = click_time + 24 * 60 * 60 * 1000
        for article_id in hot_articles['article_id'].unique():
            if article_id in past_click_articles:
                continue
            if not min_time <= hot_articles_dict[article_id] <= max_time:
                continue
            topk_click.append(article_id)
            if len(topk_click) == k:
                break
        return topk_click

    train_hot_articles = pd.DataFrame(train_click_df['click_article_id'].value_counts().index.to_list(), columns=['article_id'])
    train_hot_articles = train_hot_articles.merge(articles).drop(columns=['category_id', 'words_count'])
    train_hot_articles_dict = train_hot_articles.set_index('article_id')['created_at_ts'].to_dict()

    test_hot_articles = pd.DataFrame(test_click_df['click_article_id'].value_counts().index.to_list(), columns=['article_id'])
    test_hot_articles = test_hot_articles.merge(articles).drop(columns=['category_id', 'words_count'])
    test_hot_articles_dict = test_hot_articles.set_index('article_id')['created_at_ts'].to_dict()
    
    train_list = []
    for user_id in tqdm(train_past_clicks['user_id'].unique()):
        user = train_past_clicks.loc[train_past_clicks['user_id'] == user_id]
#         user = user[:(len(user) - 1)]
        click_time = train_last_click_time[user_id]
        past_click_articles = user['click_article_id'].values
        item_topk_click = get_item_topk_click_(train_hot_articles, train_hot_articles_dict, click_time, past_click_articles, k=topk)
        for id in item_topk_click:
            rows = [user_id, id]
            train_list.append(rows)

    hot_train_recall = pd.DataFrame(train_list, columns=['user_id', 'article_id'])
    hot_train_recall.to_csv(save_path + 'hot_train_recall.csv', index=False)

    test_list = []
    for user_id in tqdm(test_click_df['user_id'].unique()):
        user = test_click_df.loc[test_click_df['user_id'] == user_id]
        click_time = test_last_click_time[user_id]
        past_click_articles = user['click_article_id'].values
        item_topk_click = get_item_topk_click_(test_hot_articles, test_hot_articles_dict, click_time, past_click_articles, k=topk)
        for id in item_topk_click:
            rows = [user_id, id]
            test_list.append(rows)

    hot_test_recall = pd.DataFrame(test_list, columns=['user_id', 'article_id'])
    hot_test_recall.to_csv(save_path + 'hot_test_recall.csv', index=False)

    print('Hot Recall Finished! Cost time: {}'.format(time.time() - ts))
    
#测试集召回
def get_test_recall(itemcf=False, hot=False):
    if itemcf:
        itemcf_test_recall = pd.read_csv(save_path + 'itemcf_test_recall.csv')
        itemcf_test_recall = itemcf_test_recall.rename(columns={'click_article_id': 'article_id'})

    if hot:
        hot_test_recall = pd.read_csv(save_path + 'hot_test_recall.csv')

    if itemcf:
        test = itemcf_test_recall.copy()
        if hot:
            test = test.append(hot_test_recall)

    test_recall = test.drop_duplicates(['user_id', 'article_id'])
    test_recall.to_csv(save_path + 'test_recall.csv', index=False)
    print('Test Recall Finished!')
    return test_recall

#训练集召回
def get_train_recall(itemcf=False, hot=False, train_last_click=None):
    if itemcf:
        itemcf_train_recall = pd.read_csv(save_path + 'itemcf_train_recall.csv')
        itemcf_train_recall = itemcf_train_recall.rename(columns={'click_article_id': 'article_id'})
        itemcf_train_recall = itemcf_train_recall.merge(train_last_click, on=['user_id', 'article_id'], how='left')
        itemcf_train_recall['label'] = itemcf_train_recall['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
        print('Train ItemCF RECALL:{}%'.format((itemcf_train_recall['label'].value_counts()[1]) / len(train_last_click['user_id'].unique()) * 100))

    if hot:
        hot_train_recall = pd.read_csv(save_path + 'hot_train_recall.csv')
        hot_train_recall['label'] = hot_train_recall.merge(train_last_click, on=['user_id', 'article_id'], how='left')['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
        print('Train Hot RECALL:{}%'.format((hot_train_recall['label'].value_counts()[1]) / len(train_last_click['user_id'].unique()) * 100))

    if itemcf:
        train = itemcf_train_recall.copy()
        if hot:
            train = train.append(hot_train_recall)
    
    train = train.drop_duplicates(['user_id', 'article_id'])

    train['pred_score'] = train['pred_score'].fillna(-100)
    print('Train Total RECALL:{}%'.format((train['label'].value_counts()[1]) / len(train_last_click['user_id'].unique()) * 100))
    print('Train Total Recall Finished!')
    train.to_csv(save_path + 'train_recall.csv', index=False)
    
    #负采样
    def neg_sample(train=None):
        ts = time.time()

        def neg_sample_recall_data(recall_items_df, sample_rate=0.001):
            pos_data = recall_items_df[recall_items_df['label'] == 1]
            neg_data = recall_items_df[recall_items_df['label'] == 0]
            
            print('pos_data_num:', len(pos_data), 'neg_data_num:', len(neg_data), 'pos/neg:', len(pos_data)/len(neg_data))
            
            # 分组采样函数
            def neg_sample_func(group_df):
                neg_num = len(group_df)
                sample_num = max(int(neg_num * sample_rate), 1) # 保证最少有一个
                sample_num = min(sample_num, 5) # 保证最多不超过5个，这里可以根据实际情况进行选择
                return group_df.sample(n=sample_num, replace=False)
            
            # 对用户进行负采样，保证所有用户都在采样后的数据中
            neg_data_user_sample = neg_data.groupby('user_id', group_keys=False).apply(neg_sample_func)
            # 对文章进行负采样，保证所有文章都在采样后的数据中
            neg_data_item_sample = neg_data.groupby('article_id', group_keys=False).apply(neg_sample_func)
            
            # 将上述两种情况下的采样数据合并
            neg_data_new = neg_data_user_sample.append(neg_data_item_sample)
            # 由于上述两个操作是分开的，可能将两个相同的数据给重复选择了，所以需要对合并后的数据进行去重

            neg_data_new = neg_data_new.sort_values(['user_id', 'pred_score']).drop_duplicates(['user_id', 'article_id'], keep='last')
            
            # 将正样本数据合并
            data_new = pd.concat([pos_data, neg_data_new], ignore_index=True)
            
            return data_new
        train = neg_sample_recall_data(train)
        print('Negative Data Sample Finished! Cost time: {}'.format(time.time() - ts))
        return train

    train = neg_sample(train)

    return train

#训练和预测
def train_and_predict(itemcf=False, itemcf_topk=10, hot=False, hot_topk=10, offline=True):
    ts = time.time()
    
    if itemcf:
        train_past_clicks, train_last_click, test_last_click = itemcf_recall(itemcf_topk)
    if hot:
        hot_recall(hot_topk, train_past_clicks, test_last_click)
    train_past_clicks = train_past_clicks.groupby('user_id').agg({'click_timestamp': 'max'})
    train = get_train_recall(itemcf, hot, train_last_click)
    train = train.sort_values('user_id').drop(columns=['click_timestamp']).reset_index(drop=True)
    train = train.drop(columns=['click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type']).merge(train_last_click.drop(columns=['article_id', 'click_timestamp']))
    train = train.merge(articles, on='article_id', how='left')
    train = train.merge(train_past_clicks, on='user_id', how='left')
    train['delta_time'] = train['created_at_ts'] - train['click_timestamp']
    
    X = train.copy()
    y = train['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
    X_eval, X_off, y_eval, y_off = train_test_split(X_test, y_test, test_size=0.5, random_state=66)
    g_train = X_train.groupby(['user_id'], as_index=False).count()['label'].values
    g_eval = X_eval.groupby(['user_id'], as_index=False).count()['label'].values

    lgb_cols = ['click_environment','click_deviceGroup', 'click_os', 'click_country', 
                'click_region','click_referrer_type', 'category_id', 'created_at_ts', 'words_count', 'click_timestamp', 'delta_time']

    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                max_depth=-1, n_estimators=1000, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                learning_rate=0.01, min_child_weight=50, random_state=66, n_jobs=-1)
    lgb_ranker.fit(X_train[lgb_cols], y_train, group=g_train, eval_set=[(X_eval[lgb_cols], y_eval)], eval_group=[g_eval], early_stopping_rounds=50, verbose=False)
    
    #输出特征重要度
    def print_feature_importance(columns, scores):
        print('--------------------------------')
        result = list(zip(columns, scores))
        result.sort(key=lambda v: v[1], reverse=True)
        for col, score in result:
            print('{}: {}'.format(col, score))
        print('--------------------------------')

    print_feature_importance(lgb_cols, lgb_ranker.feature_importances_)
    
    X_off['pred_score'] = lgb_ranker.predict(X_off[lgb_cols], num_iteration=lgb_ranker.best_iteration_)
    X_off = X_off.drop(columns=['category_id', 'created_at_ts', 'words_count', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type', 'click_timestamp', 'delta_time'])
    recall_df = X_off.copy()
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 判断是不是每个用户都有5篇文章及以上
#         tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
#         assert tmp.min() >= topk

    del recall_df['pred_score'], recall_df['label']
    submit = recall_df[recall_df['rank'] <= 5].set_index(['user_id', 'rank']).unstack(-1).reset_index()
    max_article = int(recall_df['rank'].value_counts().index.max())
    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]

    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2', 
                                                3: 'article_3', 4: 'article_4', 5: 'article_5'})      
    submit = submit.fillna(-1)
    sums = 0
    for user_id in tqdm(submit['user_id'].unique()):
        user = submit.loc[submit['user_id'] == user_id]
        art_id = train_last_click.loc[train_last_click['user_id'] == user_id, 'article_id'].values[0]
        for i in range(1, max_article):
            if user['article_{}'.format(i)].values[0] == art_id:
                sums += 1 / i
    print('MRR:{}'.format(sums / len(submit['user_id'].unique())))
    
    if not offline:
        test_recall = get_test_recall(itemcf, hot)
        test_recall = test_recall.merge(test_last_click.drop(columns=['article_id']))
        test_recall = test_recall.merge(articles, on='article_id', how='left')
        test_recall['delta_time'] = test_recall['created_at_ts'] - test_recall['click_timestamp']

        test_recall['pred_score'] = lgb_ranker.predict(test_recall[lgb_cols], num_iteration=lgb_ranker.best_iteration_)

        result = test_recall.sort_values(by=['user_id', 'pred_score'], ascending=(True, False))
        
        result = result.drop(columns=['category_id', 'created_at_ts', 'words_count', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type', 'click_timestamp', 'delta_time'])
#         result.to_csv(save_path + 'test.csv', index=False)
        # 生成提交文件
        def submit_f(recall_df, topk=10, model_name=None):
            recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
            recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

            # 判断是不是每个用户都有5篇文章及以上
            tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
            assert tmp.min() >= topk

            del recall_df['pred_score']
            submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

            submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]

            # 按照提交格式定义列名
            submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2', 
                                                        3: 'article_3', 4: 'article_4', 5: 'article_5'})

            save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d-%H-%M') + '.csv'
            submit.to_csv(save_name, index=False, header=True)

        submit_f(result, topk=5, model_name='lgb_ranker')

        print('Submit Finished! Cost time: {}'.format(time.time() - ts))
        
offline = True

if __name__ == '__main__':
    if len(sys.argv) == 4:
        if sys.argv[1] == 'train':
            offline = True
        elif sys.argv[1] == 'test':
            offline = False
        if sys.argv[2] == 'itemcf':
            train_and_predict(itemcf=True, itemcf_topk=int(sys.argv[3]), offline=offline)
        elif sys.argv[2] == 'hot':
            train_and_predict(hot=True, hot_topk=int(sys.argv[3]), offline=offline)
        else:
            print('Wrong command.')
    elif len(sys.argv) == 6:
        if sys.argv[1] == 'train':
            offline = True
        elif sys.argv[1] == 'test':
            offline = False
        train_and_predict(itemcf=True, itemcf_topk=int(sys.argv[3]), hot=True, hot_topk=int(sys.argv[5]), offline=offline)
    else:
        print('Wrong command.')
