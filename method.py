# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import math
import random

import ItemCF
import SlopeOne
import UserCF

try:
    import LFM
except ImportError:
    pass
except NotImplementedError:
    pass


def generate_data_100k(k):
    """
    :param k: 数据集编号
    """
    global train, test
    train = {}
    test = {}
    for line in open("ml-100k/u%s.base" % k, "r"):
        user, item, rating, _ = line.split('\t')
        user, item, rating = int(user), int(item), int(rating)
        train.setdefault(user, {})
        train[user][item] = 1
    for line in open("ml-100k/u%s.test" % k, "r"):
        user, item, rating, _ = line.split('\t')
        user, item, rating = int(user), int(item), int(rating)
        test.setdefault(user, {})
        test[user][item] = 1
    global _n, _user_k, _item_k
    _n = 10
    _user_k = 50
    _item_k = 10


def generate_data_100k_with_rating(k):
    """
    :param k: 数据集编号
    """
    global train, test
    train = {}
    test = {}
    for line in open("ml-100k/u%s.base" % k, "r"):
        user, item, rating, _ = line.split('\t')
        user, item, rating = int(user), int(item), int(rating)
        train.setdefault(user, {})
        train[user][item] = rating
    for line in open("ml-100k/u%s.test" % k, "r"):
        user, item, rating, _ = line.split('\t')
        user, item, rating = int(user), int(item), int(rating)
        test.setdefault(user, {})
        test[user][item] = rating


def generate_data_1m(m, k, seed=0):
    """
    将用户行为数据集按照均匀分布随机分成m份
    挑选1份作为测试集
    将剩下的m-1份作为训练集
    :param m: 分割的份数
    :param k: 随机参数，0≤k<M
    :param seed: 随机seed
    """
    random.seed(seed)
    global train, test
    train = {}
    test = {}
    for line in open("ml-1m/ratings.dat", "r"):
        user, item, rating, _ = line.split('::')
        user, item, rating = int(user), int(item), int(rating)
        if random.randint(0, m) == k:
            test.setdefault(user, {})
            test[user][item] = 1
        else:
            train.setdefault(user, {})
            train[user][item] = 1
    global _n, _user_k, _item_k
    _n = 10
    _user_k = 80
    _item_k = 10


def generate_data_1m_with_rating(m, k, seed=0):
    """
    将用户行为数据集按照均匀分布随机分成m份
    挑选1份作为测试集
    将剩下的m-1份作为训练集
    :param m: 分割的份数
    :param k: 随机参数，0≤k<M
    :param seed: 随机seed
    """
    random.seed(seed)
    global train, test
    train = {}
    test = {}
    for line in open("ml-1m/ratings.dat", "r"):
        user, item, rating, _ = line.split('::')
        user, item, rating = int(user), int(item), int(rating)
        if random.randint(0, m) == k:
            test.setdefault(user, {})
            test[user][item] = rating
        else:
            train.setdefault(user, {})
            train[user][item] = rating


def generate_matrix(with_rating=False):
    """
    :param with_rating: 训练集是否包括rating，True则rating范围为1~5，否则为0或1
    """
    # UserCF.user_similarity_jaccard(train, iif=False)  # with/without rating
    # UserCF.user_similarity_jaccard(train, iif=True)  # with/without rating
    # UserCF.user_similarity_cosine(train, iif=False)  # with/without rating
    # UserCF.user_similarity_cosine(train, iif=True)  # with/without rating
    # UserCF.user_similarity_pearson(train, iif=False)  # with rating
    # UserCF.user_similarity_pearson(train, iif=True)  # with rating
    # UserCF.user_similarity_log_likelihood(train)  # without rating
    # ItemCF.item_similarity_jaccard(train, norm=False, iuf=False, with_rating=with_rating)  # with/without rating
    # ItemCF.item_similarity_jaccard(train, norm=True, iuf=False, with_rating=with_rating)  # with/without rating
    # ItemCF.item_similarity_jaccard(train, norm=False, iuf=True, with_rating=with_rating)  # with/without rating
    # ItemCF.item_similarity_cosine(train, norm=False, iuf=False, with_rating=with_rating)  # with/without rating
    # ItemCF.item_similarity_cosine(train, norm=True, iuf=False, with_rating=with_rating)  # with/without rating
    # ItemCF.item_similarity_cosine(train, norm=False, iuf=True, with_rating=with_rating)  # with/without rating
    # ItemCF.item_similarity_adjusted_cosine(train, iuf=False)  # with rating
    # ItemCF.item_similarity_adjusted_cosine(train, iuf=True)  # with rating
    # ItemCF.item_similarity__log_likelihood(train, norm=False)  # without rating
    # ItemCF.item_similarity__log_likelihood(train, norm=True)  # without rating
    # SlopeOne.item_deviation(train)  # with rating
    # LFM.factorization(train, bias=True, svd=False, step=100, gamma=0.01, slow_rate=0.99, Lambda=0.1)  # with rating
    LFM.factorization(train, bias=False, svd=True, step=50, gamma=0.04, slow_rate=0.93, Lambda=0.1, k=15)  # with rating
    # LFM.factorization(train, bias=True, svd=True, step=25, gamma=0.04, slow_rate=0.93, Lambda=0.1, k=15)  # with rating


def get_recommendation(user):
    # return UserCF.recommend(user, train, _n, _user_k)
    return ItemCF.recommend(user, train, _n, _item_k)


def get_recommendation_with_rating(user):
    # return UserCF.recommend_with_rating(user, train)
    # return ItemCF.recommend_with_rating(user, train)
    # return SlopeOne.recommend_with_rating(user, train)
    return LFM.recommend_with_rating(user, train)


"""
对用户u推荐n个物品，记为R(u)
令用户u在测试集上喜欢的物品集合为T(u)
"""


def recall():
    """
    召回率描述有多少比例的用户-物品评分记录包含在最终的推荐列表中
    Recall = ∑|R(u) ∩ T(u)| / ∑|T(u)|
    :return: 召回率
    """
    hit = 0
    count = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = get_recommendation(user)
        for item, pui in rank:
            if item in tu:
                hit += 1
        count += len(tu)
    return hit / count


def precision():
    """
    准确率描述最终的推荐列表中有多少比例是发生过的用户-物品评分记录
    Precision = ∑|R(u) ∩ T(u)| / ∑|R(u)|
    :return: 准确率
    """
    hit = 0
    count = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = get_recommendation(user)
        for item, pui in rank:
            if item in tu:
                hit += 1
        count += len(rank)
    return hit / count


def coverage():
    """
    该覆盖率表示最终的推荐列表中包含多大比例的物品
    Coverage = U|R(u)| / |I|
    覆盖率反映了推荐算法发掘长尾的能力，覆盖率越高，说明推荐算法越能够将长尾中的物品推荐给用户
    :return: 覆盖率
    """
    recommend_items = set()
    all_items = set()
    for user in train.iterkeys():
        for item in train[user].iterkeys():
            all_items.add(item)
        rank = get_recommendation(user)
        for item, pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / len(all_items)


def popularity():
    """
    这里用推荐列表中物品的平均流行度度量推荐结果的新颖度
    如果推荐出的物品都很热门，说明推荐的新颖度较低，否则说明推荐结果比较新颖
    计算平均流行度时对每个物品的流行度取对数，这是因为物品的流行度分布满足长尾分布，在取对数后，流行度的平均值更加稳定
    :return: 平均流行度
    """
    item_popularity = {}
    for items in train.itervalues():
        for item in items.iterkeys():
            item_popularity.setdefault(item, 0)
            item_popularity[item] += 1
    popularity_sum = 0
    count = 0
    for user in train.iterkeys():
        rank = get_recommendation(user)
        for item, pui in rank:
            popularity_sum += math.log(1 + item_popularity[item])
        count += len(rank)
    return popularity_sum / count


def RMSE():
    """
    :return: 均方根误差
    """
    rmse_sum = 0
    hit = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = get_recommendation(user)
        for item, pui in rank:
            if item in tu:
                rmse_sum += (tu[item] - pui) ** 2
                hit += 1
    return math.sqrt(rmse_sum / hit)


def MAE():
    """
    :return: 平均绝对误差
    """
    mae_sum = 0
    hit = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = get_recommendation(user)
        for item, pui in rank:
            if item in tu:
                mae_sum += abs(tu[item] - pui)
                hit += 1
    return mae_sum / hit


# def MAP():
#     """
#     :return: 平均准确率
#     """
#     map_sum = 0
#     for user in train.iterkeys():
#         hit = 0
#         count = 0
#         tu = test.get(user, {})
#         rank = get_recommendation(user)
#         for index, item in enumerate(rank):
#             if item[0] in tu:
#                 hit += 1
#                 count += hit / (index + 1)
#         map_sum += count / len(rank)
#     return map_sum / len(train)


def evaluate():
    item_popularity = {}
    for items in train.itervalues():
        for item in items.iterkeys():
            item_popularity.setdefault(item, 0)
            item_popularity[item] += 1
    hit = 0
    test_count = 0
    recommend_count = 0
    recommend_items = set()
    all_items = set()
    popularity_sum = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = get_recommendation(user)
        for item, pui in rank:
            if item in tu:
                hit += 1
            recommend_items.add(item)
            popularity_sum += math.log(1 + item_popularity[item])
        test_count += len(tu)
        recommend_count += len(rank)
        for item in train[user].iterkeys():
            all_items.add(item)
    recall_value = hit / test_count
    precision_value = hit / recommend_count
    coverage_value = len(recommend_items) / len(all_items)
    popularity_value = popularity_sum / recommend_count
    return recall_value, precision_value, coverage_value, popularity_value


def evaluate_with_rating():
    hit = 0
    rmse_sum = 0
    mae_sum = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = get_recommendation_with_rating(user)
        for item, pui in rank:
            if item in tu:
                hit += 1
                rmse_sum += (tu[item] - pui) ** 2
                mae_sum += abs(tu[item] - pui)
    rmse_value = math.sqrt(rmse_sum / hit)
    mae_value = mae_sum / hit
    return rmse_value, mae_value
