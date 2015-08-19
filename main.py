# -*- coding: utf-8 -*-

from __future__ import division

import datetime

import method

if __name__ == '__main__':
    start = datetime.datetime.now()
    ans = [0, 0, 0, 0]
    for k in xrange(0, 8):
        method.generate_data("ml-1m/ratings.dat", 8, k, 0)
        b = method.evaluate()
        for x in xrange(0, 4):
            ans[x] += b[x]
    for x in xrange(0, 4):
        ans[x] /= 8
    print ans
    end = datetime.datetime.now()
    print datetime.timedelta(seconds=(end - start).total_seconds() / 8)
