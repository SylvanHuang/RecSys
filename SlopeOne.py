# -*- coding: utf-8 -*-

from __future__ import division


def item_deviation(train):
    """
    计算物品i和j的差值
    :param train: 训练集
    """
    deviation = {}
    global _freq
    _freq = {}
    for items in train.itervalues():
        for i, ri in items.iteritems():
            _freq.setdefault(i, {})
            deviation.setdefault(i, {})
            for j, rj in items.iteritems():
                if i == j:
                    continue
                deviation[i].setdefault(j, 0)
                deviation[i][j] += ri - rj
                _freq[i].setdefault(j, 0)
                _freq[i][j] += 1
    global _w
    _w = {}
    for i, related_items in deviation.iteritems():
        _w[i] = {}
        for j, dij in related_items.iteritems():
            _w[i][j] = dij / _freq[i][j]


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
        for i, wji in _w[j].iteritems():
            if i in ru:
                continue
            rank.setdefault(i, 0)
            rank[i] += (ruj - wji) * _freq[j][i]  # wij == -wji
            freq_sum.setdefault(i, 0)
            freq_sum[i] += _freq[j][i]  # _freq[i][j] == _freq[j][i]，但后者对cache更友好
    for item in rank.iterkeys():
        if freq_sum[item]:
            rank[item] /= freq_sum[item]
    return rank.iteritems()
