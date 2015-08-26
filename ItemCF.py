# -*- coding: utf-8 -*-

from __future__ import division

import heapq
import math
import operator


def item_similarity_cosine(train, with_rating=False, norm=False, iuf=False):
    """
    通过余弦相似度计算物品i和j的相似度
    :param train: 训练集
    """
    c = {}
    n = {}
    for items in train.itervalues():
        for i, ri in items.iteritems():
            n.setdefault(i, 0)
            n[i] += ri ** 2
            c.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                c[i].setdefault(j, 0)
                c[i][j] += ri * rj if not iuf else ri * rj / math.log(1 + len(items))
    global w
    w = {}
    for i, related_items in c.iteritems():
        w[i] = []
        for j, cij in related_items.iteritems():
            w[i].append((j, cij / math.sqrt(n[i] * n[j])))
        if norm:
            wmax = max(wij for _, wij in w[i])
            for item in w[i]:
                item[1] /= wmax
        if not with_rating:
            w[i].sort(key=operator.itemgetter(1), reverse=True)


def recommend(user, n, train, k):
    """
    用户u对物品i评分的可能性预测
    :param user: 用户
    :param train: 训练集
    :param n: 为用户推荐n个物品
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


def recommend_with_rating(user, train):
    """
    用户u对物品i的评分预测
    :param user: 用户
    :param train: 训练集
    :return: 推荐列表
    """
    rank = {}
    w_sum = {}
    ru = train[user]
    for j, ruj in ru.iteritems():
        for i, wij in w[j]:
            if i in ru:
                continue
            rank.setdefault(i, 0)
            rank[i] += wij * ruj
            w_sum.setdefault(i, 0)
            w_sum[i] += wij
    for item in rank.iterkeys():
        rank[item] /= w_sum[item]
    return rank.items()


def get_nr(r):
    return (r - 1) / 2 - 1


def get_r(nr):
    return (nr + 1) * 2 + 1
