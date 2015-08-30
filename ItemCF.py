# -*- coding: utf-8 -*-

from __future__ import division

import heapq
import math
import operator


def item_similarity_jaccard(train, norm=False, iuf=False, with_rating=False):
    """
    通过Jaccard相似度计算物品i和j的相似度
    :param train: 训练集
    """
    global avr
    avr = {}
    c = {}
    n = {}
    for user, items in train.iteritems():
        item_len = len(items)
        avr[user] = 0  # Jaccard相似度不需要计算偏移
        for i, ri in items.iteritems():
            n.setdefault(i, 0)
            n[i] += ri ** 2
            c.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                c[i].setdefault(j, 0)
                c[i][j] += ri * rj if not iuf else ri * rj / math.log(1 + item_len)
    global w
    w = {}
    for i, related_items in c.iteritems():
        w[i] = []
        for j, cij in related_items.iteritems():
            w[i].append([j, cij / (n[i] + n[j] - cij)])
        if norm:
            wmax = max(item[1] for item in w[i])
            for item in w[i]:
                item[1] /= wmax
        if not with_rating:
            w[i].sort(key=operator.itemgetter(1), reverse=True)


def item_similarity_cosine(train, norm=False, iuf=False, with_rating=False):
    """
    通过余弦相似度计算物品i和j的相似度
    :param train: 训练集
    """
    global avr
    avr = {}
    c = {}
    n = {}
    for user, items in train.iteritems():
        item_len = len(items)
        avr[user] = 0  # 余弦相似度不需要计算偏移
        for i, ri in items.iteritems():
            n.setdefault(i, 0)
            n[i] += ri ** 2
            c.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                c[i].setdefault(j, 0)
                c[i][j] += ri * rj if not iuf else ri * rj / math.log(1 + item_len)
    global w
    w = {}
    for i, related_items in c.iteritems():
        w[i] = []
        for j, cij in related_items.iteritems():
            w[i].append([j, cij / math.sqrt(n[i] * n[j])])
        if norm:
            wmax = max(item[1] for item in w[i])
            for item in w[i]:
                item[1] /= wmax
        if not with_rating:
            w[i].sort(key=operator.itemgetter(1), reverse=True)


def item_similarity_adjusted_cosine(train, iuf=False):
    """
    通过余弦相似度计算物品i和j的相似度
    :param train: 训练集
    """
    global avr
    avr = {}
    c = {}
    n = {}
    for user, items in train.iteritems():
        item_len = len(items)
        avr[user] = sum(items.itervalues()) / item_len
        for i, ri in items.iteritems():
            n.setdefault(i, 0)
            n[i] += (ri - avr[user]) ** 2
            c.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                c[i].setdefault(j, 0)
                c[i][j] += (ri - avr[user]) * (rj - avr[user]) if not iuf else (ri - avr[user]) * (
                    rj - avr[user]) / math.log(1 + item_len)
    global w
    w = {}
    for i, related_items in c.iteritems():
        w[i] = []
        for j, cij in related_items.iteritems():
            w[i].append([j, cij / math.sqrt(n[i] * n[j]) if n[i] * n[j] else 0])


def item_similarity__log_likelihood(train, norm=False):
    """
    通过对数似然比计算物品i和j的相似度
    :param train: 训练集
    """
    c = {}
    n = {}
    for user, items in train.iteritems():
        for i, ri in items.iteritems():
            n.setdefault(i, 0)
            n[i] += ri ** 2
            c.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                c[i].setdefault(j, 0)
                c[i][j] += ri * rj
    global w
    w = {}
    user_len = len(train)
    for i, related_items in c.iteritems():
        w[i] = []
        for j, cij in related_items.iteritems():
            w[i].append([j, __calc_log_likelihood(cij, n[i] - cij, n[j] - cij, user_len - n[i] - n[j] + cij)])
        if norm:
            wmax = max(item[1] for item in w[i])
            for item in w[i]:
                item[1] /= wmax
        w[i].sort(key=operator.itemgetter(1), reverse=True)


def __calc_log_likelihood(num_both, num_x, num_y, num_none):
    """
    :param num_both: x和y共同偏好的数量
    :param num_x: x单独偏好的数量
    :param num_y: y单独偏好的数量
    :param num_none: x和y都不偏好的数量
    :return: 对数似然比
    """
    p1 = num_both / (num_both + num_x)
    p2 = num_y / (num_y + num_none)
    p = (num_both + num_y) / (num_both + num_x + num_y + num_none)
    r1 = 0
    r2 = 0
    if 0 < p <= 1:
        r1 += num_both * math.log(p) + num_y * math.log(p)
    if 0 <= p < 1:
        r1 += num_x * math.log(1 - p) + num_none * math.log(1 - p)
    if 0 < p1 <= 1:
        r2 += num_both * math.log(p1)
    if 0 <= p1 < 1:
        r2 += num_x * math.log(1 - p1)
    if 0 < p2 <= 1:
        r2 += num_y * math.log(p2)
    if 0 <= p2 < 1:
        r2 += num_none * math.log(1 - p2)
    return 2 * (r2 - r1)


def recommend(user, train, n, k):
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
        for i, wji in w[j][:k]:
            if i in ru:
                continue
            rank.setdefault(i, 0)
            rank[i] += wji * ruj  # wij == wji
            # rank.setdefault(i, {})
            # rank[i].setdefault("weight", 0)
            # rank[i]["weight"] += ruj * wji
            # rank[i].setdefault("reason", {})
            # rank[i]["reason"][j] = ruj * wji
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
        for i, wji in w[j]:
            if i in ru:
                continue
            rank.setdefault(i, 0)
            rank[i] += wji * (ruj - avr[user])  # wij == wji
            w_sum.setdefault(i, 0)
            w_sum[i] += abs(wji)
    for item in rank.iterkeys():
        if w_sum[item]:
            rank[item] /= w_sum[item]
        rank[item] += avr[user]
    return rank.iteritems()
