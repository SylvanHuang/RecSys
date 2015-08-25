# -*- coding: utf-8 -*-

from __future__ import division

import heapq
import math
import operator


def user_similarity(train):
    """
    通过余弦相似度计算u和v的兴趣相似度
    W(u,v) = |N(u) ∩ N(v)| / sqrt(|N(u)||N(v)|)
    :param train: 训练集
    :return: 用户相似度矩阵
    """
    # build inverse table for item_users
    item_users = {}
    for user, items in train.iteritems():
        for item, rating in items.iteritems():
            item_users.setdefault(item, {})
            item_users[item][user] = rating
    # calculate co-rated items between users
    c = {}
    n = {}
    for users in item_users.itervalues():
        for u, ru in users.iteritems():
            n.setdefault(u, 0)
            n[u] += ru ** 2
            c.setdefault(u, {})
            for v, rv in users.iteritems():
                if u == v:
                    continue
                c[u].setdefault(v, 0)
                c[u][v] += ru * rv
    # calculate finial similarity matrix W
    w = {}
    for u, related_users in c.iteritems():
        w[u] = {}
        for v, cuv in related_users.iteritems():
            w[u][v] = cuv / math.sqrt(n[u] * n[v])
    return w


def user_similarity_iif(train):
    """
    计算u和v的兴趣相似度，惩罚了用户u和用户v共同兴趣列表中热门物品对他们相似度的影响
    W(u,v) = ∑(1 / log(1 + |N(i)|)) / sqrt(|N(u)||N(v)|), i ∈ |N(u) ∩ N(v)|
    :param train: 训练集
    :return: 用户相似度矩阵
    """
    # build inverse table for item_users
    item_users = {}
    for user, items in train.iteritems():
        for item, rating in items.iteritems():
            item_users.setdefault(item, {})
            item_users[item][user] = rating
    # calculate co-rated items between users
    c = {}
    n = {}
    for users in item_users.itervalues():
        for u, ru in users.iteritems():
            n.setdefault(u, 0)
            n[u] += ru ** 2
            c.setdefault(u, {})
            for v, rv in users.iteritems():
                if u == v:
                    continue
                c[u].setdefault(v, 0)
                c[u][v] += ru * rv / math.log(1 + len(users))
    # calculate finial similarity matrix W
    w = {}
    for u, related_users in c.iteritems():
        w[u] = {}
        for v, cuv in related_users.iteritems():
            w[u][v] = cuv / math.sqrt(n[u] * n[v])
    return w


def recommend(user, n, train, w, k):
    """
    用户u对物品i的感兴趣程度
    p(u,i) = ∑(W(u,v) * R(v,i)), v ∈ S(u,k) ∩ N(i)
    其中S(u,k)包含和用户u兴趣最接近的k个用户，N(i)是对物品i有过行为的用户集合，W(u,v)是用户u和用户v的兴趣相似度
    R(v,i)代表用户v对物品i的兴趣，因为使用的是单一行为的隐反馈数据，所以所有的rvi=1
    :param user: 用户
    :param train: 训练集
    :param n: 为用户推荐n个物品
    :param w: 用户相似度矩阵
    :param k: 取和用户u兴趣最接近的k个用户
    :return: 推荐列表
    """
    rank = {}
    ru = train[user]
    for v, wuv in heapq.nlargest(k, w[user].iteritems(), key=operator.itemgetter(1)):
        for i, rvi in train[v].iteritems():
            if i in ru:
                # we should filter items user interacted before
                continue
            rank.setdefault(i, 0)
            rank[i] += wuv * rvi
    return heapq.nlargest(n, rank.iteritems(), key=operator.itemgetter(1))


def recommend_with_rating(user, train, w):
    rank = {}
    w_sum = {}
    ru = train[user]
    for v, wuv in w[user].iteritems():
        for i, rvi in train[v].iteritems():
            if i in ru:
                # we should filter items user interacted before
                continue
            rank.setdefault(i, 0)
            rank[i] += wuv * rvi
            w_sum.setdefault(i, 0)
            w_sum[i] += wuv
    for item in rank.iterkeys():
        rank[item] /= w_sum[item]
    return rank.items()
