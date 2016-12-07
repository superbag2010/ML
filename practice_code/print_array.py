#! /usr/bin/python3

import csv
import numpy as np
import tensorflow as tf
import os

class student:
	a = 1
	b = 2


array = [[1, 2, 3], [4, 5, 6]]

print(dir(student))


for i, name in enumerate(['first', 'second', 'third']):
	print (i, name)

array = np.array(array)

print(array[:2])
print(array.shape) # print shape

a = [None, 1]
print(a)

print("===============================")

ls = [None] * 10

for j, l in enumerate(ls):
    ls[j] = [j, j, j]
    print ("ls[{}] = {}".format(j, ls[j]))

integer = 0
ls[integer + 1] = 100
print(ls[1])


print("==============================")
tf.flags.DEFINE_string("ff", "3,0,4", "test")
FLAGS = tf.flags.FLAGS
b = list(map(int, FLAGS.ff.split(",")))
for k, v in enumerate(b):
    if v == 0: break
    print("b[{}] = {}".format(k, v))
print("b[{}] = {}".format(k, b[k]))

print("=============================")
c = [None]*10
c[3] = 100
print(c[-1])


print("=============================")
t = ["character", "number"]
m = np.column_stack((["charactor"] + ['a', 'b'], ["number"] + ['1', '2']))
print(m)
out_path = os.path.join("./", "test.csv")
#with open(out_path, 'w') as f:
#    csv.writer(f).writerows(t)
