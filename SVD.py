# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy


def train_matrix_factorization(train, k=30, step=25, gamma=0.04, Lambda=0.15):
    global bu, bi, pu, qi, _k, movies
    bu = {}
    bi = {}
    pu = {}
    qi = {}
    _k = k
    mat = []
    movies = set()
    for user, items in train.iteritems():
        bu.setdefault(user, 0)
        pu.setdefault(user, numpy.random.random((_k, 1)) / 10 * (numpy.sqrt(_k)))
        for item, rating in items.iteritems():
            mat.append((user, item, rating))
            movies.add(item)
            bi.setdefault(item, 0)
            qi.setdefault(item, numpy.random.random((_k, 1)) / 10 * (numpy.sqrt(_k)))
    mat = numpy.array(mat)
    global avr
    avr = numpy.mean(mat[:, 2])
    for _ in xrange(step):
        rmse_sum = 0
        mae_sum = 0
        to = numpy.random.permutation(mat.shape[0])
        for i in xrange(mat.shape[0]):
            user = mat[to[i]][0]
            item = mat[to[i]][1]
            rating = mat[to[i]][2]
            rui = rating - predict(user, item)
            rmse_sum += rui ** 2
            mae_sum += abs(rui)
            bu[user] += gamma * (rui - Lambda * bu[user])
            bi[item] += gamma * (rui - Lambda * bi[item])
            pu[user], qi[item] = pu[user] + gamma * (rui * qi[item] - Lambda * pu[user]), qi[item] + gamma * (
                rui * pu[user] - Lambda * qi[item])
        gamma *= 0.93
        print "step: %s, rmse: %s, mae: %s" % (_ + 1, numpy.sqrt(rmse_sum / mat.shape[0]), mae_sum / mat.shape[0])


def predict(user, item):
    bu.setdefault(user, 0)
    bi.setdefault(item, 0)
    pu.setdefault(user, numpy.zeros((_k, 1)))
    qi.setdefault(item, numpy.zeros((_k, 1)))
    rui = avr + bu[user] + bi[item] + numpy.sum(pu[user] * qi[item])
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
    for item in movies:
        if item in ru:
            continue
        rank[item] = predict(user, item)
    return rank.iteritems()
