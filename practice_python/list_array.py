#! /usr/bin/python3

import numpy as np

l = [[1], [2], [3]]
a = np.array(l)
l = np.array(l)

print("list = {}".format(l))
print("array = {}".format(a))
print("array.shape = {}".format(a.shape))

"""
print("+++++++++++++++++++++++")

l = l + [4]
print("after +, list = {}".format(l))

l.append([4])
print("after append(), list = {}".format(l))
"""
