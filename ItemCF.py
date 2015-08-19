# -*- coding: utf-8 -*-

from __future__ import division

import heapq
import math
import operator


def item_similarity(train):
    """
    通过余弦相似度计算物品i和j的相似度
    W(i,j) = |N(i) ∩ N(j)| / sqrt(|N(i)||N(j)|)
    :param train: 训练集
    :return: 物品相似度矩阵
    """
    # calculate co-rated users between items
    c = {}
    n = {}
    for items in train.itervalues():
        for i in items.iterkeys():
            n.setdefault(i, 0)
            n[i] += 1
            c.setdefault(i, {})
            for j in items.iterkeys():
                if i == j:
                    continue
                c[i].setdefault(j, 0)
                c[i][j] += 1
    # calculate finial similarity matrix W
    w = {}
    for i, related_items in c.iteritems():
        w[i] = []
        for j, cij in related_items.iteritems():
            w[i].append((j, cij / math.sqrt(n[i] * n[j])))
        w[i].sort(key=operator.itemgetter(1), reverse=True)
    return w


def item_similarity_norm(train):
    """
    通过余弦相似度计算物品i和j的相似度，并进行归一化
    W(i,j) = |N(i) ∩ N(j)| / sqrt(|N(i)||N(j)|)
    W(i,j) = W(i,j) / max(W(i))
    :param train: 训练集
    :return: 物品相似度矩阵
    """
    # calculate co-rated users between items
    c = {}
    n = {}
    for items in train.itervalues():
        for i in items.iterkeys():
            n.setdefault(i, 0)
            n[i] += 1
            c.setdefault(i, {})
            for j in items.iterkeys():
                if i == j:
                    continue
                c[i].setdefault(j, 0)
                c[i][j] += 1
    # calculate finial similarity matrix W
    w = {}
    for i, related_items in c.iteritems():
        w[i] = []
        for j, cij in related_items.iteritems():
            w[i].append([j, cij / math.sqrt(n[i] * n[j])])
        w[i].sort(key=operator.itemgetter(1), reverse=True)
        for item in w[i]:
            item[1] /= w[i][0][1]
    return w


def item_similarity_iuf(train):
    """
    计算物品i和j的相似度，对活跃用户的兴趣列表带来的相似度贡献进行了惩罚
    W(i,j) = ∑(1 / log(1 + |N(u)|)) / sqrt(|N(i)||N(j)|), u ∈ |N(i) ∩ N(j)|
    :param train: 训练集
    :return: 物品相似度矩阵
    """
    # calculate co-rated users between items
    c = {}
    n = {}
    for items in train.itervalues():
        for i in items.iterkeys():
            n.setdefault(i, 0)
            n[i] += 1
            c.setdefault(i, {})
            for j in items.iterkeys():
                if i == j:
                    continue
                c[i].setdefault(j, 0)
                c[i][j] += 1 / math.log(1 + len(items))
    # calculate finial similarity matrix W
    w = {}
    for i, related_items in c.iteritems():
        w[i] = []
        for j, cij in related_items.iteritems():
            w[i].append((j, cij / math.sqrt(n[i] * n[j])))
        w[i].sort(key=operator.itemgetter(1), reverse=True)
    return w


def recommend(user, n, train, w, k):
    """
    用户u对物品i的感兴趣程度
    p(u,i) = ∑(W(i,j) * R(u,j)), j ∈ S(i,k) ∩ N(u)
    其中S(i,k)是和物品i最相似的k个物品的集合，N(u)是用户喜欢的物品的集合，W(i,j)是物品i和j的相似度
    R(u,j)代表用户u对物品j的兴趣，因为使用的是单一行为的隐反馈数据，所以所有的ruj=1
    :param user: 用户
    :param train: 训练集
    :param n: 为用户推荐n个物品
    :param w: 物品相似度矩阵
    :param k: 取和物品j最相似的k个物品
    :return: 推荐列表
    """
    rank = {}
    ru = train[user]
    for j, ruj in ru.iteritems():
        for i, wij in w[j][:k]:
            if i in ru:
                # we should filter items user interacted before
                continue
            rank.setdefault(i, 0)
            rank[i] += wij * ruj
            # rank.setdefault(i, {})
            # rank[i].setdefault("weight", 0)
            # rank[i]["weight"] += ruj * wij
            # rank[i].setdefault("reason", {})
            # rank[i]["reason"][j] = ruj * wij
    return heapq.nlargest(n, rank.iteritems(), key=operator.itemgetter(1))
