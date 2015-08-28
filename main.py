# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import datetime

import method


def test1m():
    start = datetime.datetime.now()
    ans = [0, 0, 0, 0]
    for k in xrange(0, 8):
        method.generate_data_1m(8, k, 0)
        method.generate_matrix()
        b = method.evaluate()
        for x in xrange(0, 4):
            ans[x] += b[x]
    for x in xrange(0, 4):
        ans[x] /= 8
    print ans
    end = datetime.datetime.now()
    print datetime.timedelta(seconds=(end - start).total_seconds() / 8)


def test100k():
    start = datetime.datetime.now()
    ans = [0, 0, 0, 0]
    for k in xrange(1, 6):
        method.generate_data_100k(k)
        method.generate_matrix()
        b = method.evaluate()
        for x in xrange(0, 4):
            ans[x] += b[x]
    for x in xrange(0, 4):
        ans[x] /= 5
    print ans
    end = datetime.datetime.now()
    print datetime.timedelta(seconds=(end - start).total_seconds() / 5)


def test100k_with_rating():
    start = datetime.datetime.now()
    ans = [0, 0]
    for k in xrange(1, 6):
        method.generate_data_100k_with_rating(k)
        method.generate_matrix(with_rating=True)
        b = method.evaluate_with_rating()
        for x in xrange(0, 2):
            ans[x] += b[x]
    for x in xrange(0, 2):
        ans[x] /= 5
    print ans
    end = datetime.datetime.now()
    print datetime.timedelta(seconds=(end - start).total_seconds() / 5)


if __name__ == '__main__':
    # test1m()
    # test100k()
    test100k_with_rating()
