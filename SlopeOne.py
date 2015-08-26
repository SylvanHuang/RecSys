# -*- coding: utf-8 -*-

from __future__ import division

frequency = {}


def item_deviation(train):
    """
    计算物品i和j的差值
    :param train: 训练集
    """
    deviation = {}
    # 获取每位用户的评分数据
    for items in train.itervalues():
        # 对于该用户的每个评分项（歌手、分数）
        for i, ri in items.iteritems():
            frequency.setdefault(i, {})
            deviation.setdefault(i, {})
            # 再次遍历该用户的每个评分项
            for j, rj in items.iteritems():
                if i == j:
                    continue
                # 将评分的差异保存到变量中
                deviation[i].setdefault(j, 0)
                deviation[i][j] += ri - rj
                frequency[i].setdefault(j, 0)
                frequency[i][j] += 1
    global w
    w = {}
    for i, related_items in deviation.iteritems():
        w[i] = {}
        for j, dij in related_items.iteritems():
            w[i][j] = dij / frequency[i][j]


def recommend_with_rating(user, train):
    """
    用户u对物品i的评分预测
    :param user: 用户
    :param train: 训练集
    :return: 推荐列表
    """
    rank = {}
    ru = train[user]
    for i, wi in w.iteritems():
        if i not in ru:
            rank[i] = 0
            freq = 0
            for j, ruj in ru.iteritems():
                if j in wi:
                    rank[i] += (wi[j] + ruj) * frequency[i][j]
                    freq += frequency[i][j]
            if freq:
                rank[i] /= freq
    return rank.items()
