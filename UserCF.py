# -*- coding: utf-8 -*-

from __future__ import division

import heapq
import math
import operator


def user_similarity_jaccard(train, iif=False):
    """
    通过Jaccard相似度计算u和v的兴趣相似度
    :param train: 训练集
    """
    global avr
    avr = {}
    item_users = {}
    for user, items in train.iteritems():
        avr[user] = 0  # Jarccard相似度不需要计算偏移
        for item, rating in items.iteritems():
            item_users.setdefault(item, {})
            item_users[item][user] = rating
    c = {}
    n = {}
    for users in item_users.itervalues():
        user_len = len(users)
        for u, ru in users.iteritems():
            n.setdefault(u, 0)
            n[u] += ru ** 2
            c.setdefault(u, {})
            for v, rv in users.iteritems():
                if u == v:
                    continue
                c[u].setdefault(v, 0)
                c[u][v] += ru * rv if not iif else ru * rv / math.log(1 + user_len)
    global w
    w = {}
    for u, related_users in c.iteritems():
        w[u] = {}
        for v, cuv in related_users.iteritems():
            w[u][v] = cuv / (n[u] + n[v] - cuv)


def user_similarity_cosine(train, iif=False):
    """
    通过余弦相似度计算u和v的兴趣相似度
    :param train: 训练集
    """
    global avr
    avr = {}
    item_users = {}
    for user, items in train.iteritems():
        avr[user] = 0  # 余弦相似度不需要计算偏移
        for item, rating in items.iteritems():
            item_users.setdefault(item, {})
            item_users[item][user] = rating
    c = {}
    n = {}
    for users in item_users.itervalues():
        user_len = len(users)
        for u, ru in users.iteritems():
            n.setdefault(u, 0)
            n[u] += ru ** 2
            c.setdefault(u, {})
            for v, rv in users.iteritems():
                if u == v:
                    continue
                c[u].setdefault(v, 0)
                c[u][v] += ru * rv if not iif else ru * rv / math.log(1 + user_len)
    global w
    w = {}
    for u, related_users in c.iteritems():
        w[u] = {}
        for v, cuv in related_users.iteritems():
            w[u][v] = cuv / math.sqrt(n[u] * n[v])


def user_similarity_pearson(train, iif=False):
    """
    通过皮尔逊相关系数计算u和v的兴趣相似度
    :param train: 训练集
    """
    global avr
    avr = {}
    item_users = {}
    for user, items in train.iteritems():
        avr[user] = sum(items.itervalues()) / len(items)
        for item, rating in items.iteritems():
            item_users.setdefault(item, {})
            item_users[item][user] = rating
    avr_x = {}
    avr_y = {}
    tot = {}
    for users in item_users.itervalues():
        for u, ru in users.iteritems():
            avr_x.setdefault(u, {})
            avr_y.setdefault(u, {})
            tot.setdefault(u, {})
            for v, rv in users.iteritems():
                if u == v:
                    continue
                avr_x[u].setdefault(v, 0)
                avr_x[u][v] += ru
                avr_y[u].setdefault(v, 0)
                avr_y[u][v] += rv
                tot[u].setdefault(v, 0)
                tot[u][v] += 1
    for u, related_users in tot.iteritems():
        for v, cnt in related_users.iteritems():
            avr_x[u][v] /= cnt
            avr_y[u][v] /= cnt
    c = {}
    x = {}
    y = {}
    for users in item_users.itervalues():
        user_len = len(users)
        for u, ru in users.iteritems():
            c.setdefault(u, {})
            x.setdefault(u, {})
            y.setdefault(u, {})
            for v, rv in users.iteritems():
                if u == v:
                    continue
                c[u].setdefault(v, 0)
                c[u][v] += (ru - avr_x[u][v]) * (rv - avr_y[u][v]) if not iif else (ru - avr_x[u][v]) * (
                    rv - avr_y[u][v]) / math.log(1 + user_len)
                x[u].setdefault(v, 0)
                x[u][v] += (ru - avr_x[u][v]) ** 2
                y[u].setdefault(v, 0)
                y[u][v] += (rv - avr_y[u][v]) ** 2
    global w
    w = {}
    for u, related_users in c.iteritems():
        w[u] = {}
        for v, cuv in related_users.iteritems():
            w[u][v] = cuv / math.sqrt(x[u][v] * y[u][v]) if x[u][v] * y[u][v] else 0


def user_similarity_log_likelihood(train):
    """
    通过对数似然比计算u和v的兴趣相似度
    :param train: 训练集
    """
    item_users = {}
    for user, items in train.iteritems():
        for item, rating in items.iteritems():
            item_users.setdefault(item, {})
            item_users[item][user] = rating
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
    global w
    w = {}
    item_len = len(item_users)
    for u, related_users in c.iteritems():
        w[u] = {}
        for v, cuv in related_users.iteritems():
            w[u][v] = __calc_log_likelihood(cuv, n[u] - cuv, n[v] - cuv, item_len - n[u] - n[v] + cuv)


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
    :param k: 取和用户u兴趣最接近的k个用户
    :return: 推荐列表
    """
    rank = {}
    ru = train[user]
    for v, wuv in heapq.nlargest(k, w[user].iteritems(), key=operator.itemgetter(1)):
        for i, rvi in train[v].iteritems():
            if i in ru:
                continue
            rank.setdefault(i, 0)
            rank[i] += wuv * rvi
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
    for v, wuv in w[user].iteritems():
        for i, rvi in train[v].iteritems():
            if i in ru:
                continue
            rank.setdefault(i, 0)
            rank[i] += wuv * (rvi - avr[v])
            w_sum.setdefault(i, 0)
            w_sum[i] += abs(wuv)
    for item in rank.iterkeys():
        if w_sum[item]:
            rank[item] /= w_sum[item]
        rank[item] += avr[user]
    return rank.iteritems()
