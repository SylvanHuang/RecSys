# -*- coding: utf-8 -*-

from __future__ import division


def item_deviation(train):
    """
    计算物品i和j的差值
    :param train: 训练集
    """
    deviation = {}
    global freq
    freq = {}
    for items in train.itervalues():
        for i, ri in items.iteritems():
            freq.setdefault(i, {})
            deviation.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                deviation[i].setdefault(j, 0)
                deviation[i][j] += ri - rj
                freq[i].setdefault(j, 0)
                freq[i][j] += 1
    global w
    w = {}
    for i, related_items in deviation.iteritems():
        w[i] = {}
        for j, dij in related_items.iteritems():
            w[i][j] = dij / freq[i][j]


def recommend_with_rating(user, train):
    """
    用户u对物品i的评分预测
    :param user: 用户
    :param train: 训练集
    :return: 推荐列表
    """
    rank = {}
    freq_sum = {}
    ru = train[user]
    for j, ruj in ru.iteritems():
        for i, wji in w[j].iteritems():
            if i in ru:
                continue
            rank.setdefault(i, 0)
            rank[i] += (ruj - wji) * freq[j][i]  # wij == -wji
            freq_sum.setdefault(i, 0)
            freq_sum[i] += freq[j][i]  # freq[i][j] == freq[j][i]，但后者对cache更友好
    for item in rank.iterkeys():
        if freq_sum[item]:
            rank[item] /= freq_sum[item]
    return rank.items()
