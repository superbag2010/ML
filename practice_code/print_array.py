#! /usr/bin/python3

import numpy as np

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
