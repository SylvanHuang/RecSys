# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy


def train_matrix_factorization(train, k=30, step=25, gamma=0.04, Lambda=0.15):
    global _k
    _k = k
    global _bu, _bi, _pu, _qi, _movies, _avr
    _bu = {}
    _bi = {}
    _pu = {}
    _qi = {}
    mat = []
    _movies = set()
    _avr = 0
    tot = 0
    for user, items in train.iteritems():
        _bu.setdefault(user, 0)
        _pu.setdefault(user, numpy.random.random((_k, 1)) * 0.1 * (numpy.sqrt(_k)))
        for item, rating in items.iteritems():
            mat.append((user, item, rating))
            _movies.add(item)
            _bi.setdefault(item, 0)
            _qi.setdefault(item, numpy.random.random((_k, 1)) * 0.1 * (numpy.sqrt(_k)))
            _avr += rating
            tot += 1
    _avr /= tot
    for _ in xrange(step):
        rmse_sum = 0
        mae_sum = 0
        for user, items in train.iteritems():
            for item, rating in items.iteritems():
                rui = rating - predict(user, item)
                _bu[user] += gamma * (rui - Lambda * _bu[user])
                _bi[item] += gamma * (rui - Lambda * _bi[item])
                _pu[user], _qi[item] = _pu[user] + gamma * (rui * _qi[item] - Lambda * _pu[user]), _qi[item] + gamma * (
                    rui * _pu[user] - Lambda * _qi[item])
                rmse_sum += rui ** 2
                mae_sum += abs(rui)
        gamma *= 0.93
        print "step: %s, rmse: %s, mae: %s" % (_ + 1, numpy.sqrt(rmse_sum / tot), mae_sum / tot)


def predict(user, item):
    _bu.setdefault(user, 0)
    _bi.setdefault(item, 0)
    _pu.setdefault(user, numpy.zeros((_k, 1)))
    _qi.setdefault(item, numpy.zeros((_k, 1)))
    rui = _avr + _bu[user] + _bi[item] + numpy.sum(_pu[user] * _qi[item])
    # if rui > 5:
    #     rui = 5
    # elif rui < 1:
    #     rui = 1
    return rui


def recommend_with_rating(user, train):
    """
    用户u对物品i的评分预测
    :param user: 用户
    :param train: 训练集
    :return: 推荐列表
    """
    rank = {}
    ru = train[user]
    for item in _movies:
        if item in ru:
            continue
        rank[item] = predict(user, item)
    return rank.iteritems()
