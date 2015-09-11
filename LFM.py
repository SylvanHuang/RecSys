# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import heapq
import operator

import numpy


def factorization(train, bias=True, svd=True, step=25, gamma=0.04, slow_rate=0.93, Lambda=0.1, seed=0, k=30,
                  with_rating=True):
    """
    建立隐语义模型，并使用随机梯度下降优化
    :param train: 训练集
    :param bias: 是否计算偏移
    :param svd: 是否使用奇异值分解
    :param step: 迭代次数
    :param gamma: 步长
    :param slow_rate: 步长减缓的系数
    :param Lambda: 正则化参数
    :param k: 奇异值分解向量长度
    """
    numpy.random.seed(seed)
    global _bias, _svd, _k
    _bias = bias
    _svd = svd
    _k = k
    global _bu, _bi, _pu, _qi, _movie_list, _movie_set, _avr, _tot
    _bu = {}
    _bi = {}
    _pu = {}
    _qi = {}
    _movie_list = []
    _avr = 0
    _tot = 0
    for user, items in train.iteritems():
        if _bias:
            _bu.setdefault(user, 0)
        if _svd:
            _pu.setdefault(user, numpy.random.random((_k, 1)) * 0.1 * (numpy.sqrt(_k)))
        for item, rating in items.iteritems():
            _movie_list.append(item)
            if _bias:
                _bi.setdefault(item, 0)
            if _svd:
                _qi.setdefault(item, numpy.random.random((_k, 1)) * 0.1 * (numpy.sqrt(_k)))
            _avr += rating
            _tot += 1
    _movie_set = set(_movie_list)
    _avr /= _tot
    for _ in xrange(step):
        rmse_sum = 0
        mae_sum = 0
        for user, items in train.iteritems():
            for item, rating in items.iteritems() if with_rating else __random_negative_sample(items).iteritems():
                rui = rating - __predict(user, item)
                if _bias:
                    _bu[user] += gamma * (rui - Lambda * _bu[user])
                    _bi[item] += gamma * (rui - Lambda * _bi[item])
                if _svd:
                    _pu[user], _qi[item] = _pu[user] + gamma * (rui * _qi[item] - Lambda * _pu[user]), _qi[
                        item] + gamma * (rui * _pu[user] - Lambda * _qi[item])
                rmse_sum += rui ** 2
                mae_sum += abs(rui)
        gamma *= slow_rate
        print "step: %s, rmse: %s, mae: %s" % (_ + 1, numpy.sqrt(rmse_sum / _tot), mae_sum / _tot)


def __random_negative_sample(items):
    ret = {}
    for item in items.iterkeys():
        ret[item] = 1
    n = 0
    items_len = len(items)
    for _ in xrange(items_len * 20):
        item = _movie_list[int(numpy.random.random() * _tot)]
        if item in ret:
            continue
        ret[item] = 0
        n += 1
        if n > items_len * 10:
            break
    return ret


def __predict(user, item):
    """
    预测用户对单件物品的评分
    :param user: 用户
    :param item: 物品
    :return: 预测值
    """
    rui = 0
    if _bias:
        _bu.setdefault(user, 0)
        _bi.setdefault(item, 0)
        rui += _avr + _bu[user] + _bi[item]
    if _svd:
        _pu.setdefault(user, numpy.zeros((_k, 1)))
        _qi.setdefault(item, numpy.zeros((_k, 1)))
        rui += numpy.sum(_pu[user] * _qi[item])
    # if rui > 5:
    #     rui = 5
    # elif rui < 1:
    #     rui = 1
    return rui


def recommend(user, train, n):
    """
    用户u对物品i的评分预测
    :param user: 用户
    :param train: 训练集
    :return: 推荐列表
    """
    rank = {}
    ru = train[user]
    for item in _movie_set:
        if item in ru:
            continue
        rank[item] = __predict(user, item)
    return heapq.nlargest(n, rank.iteritems(), key=operator.itemgetter(1))


def recommend_with_rating(user, train):
    """
    用户u对物品i的评分预测
    :param user: 用户
    :param train: 训练集
    :return: 推荐列表
    """
    rank = {}
    ru = train[user]
    for item in _movie_set:
        if item in ru:
            continue
        rank[item] = __predict(user, item)
    return rank.iteritems()
