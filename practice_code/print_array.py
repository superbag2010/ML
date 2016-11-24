import numpy as np

class student:
	a = 1
	b = 2


array = [1, 2, 3]

print(dir(student))


for i, name in enumerate(['first', 'second', 'third']):
	print (i, name)


print ("input a number you want to print")

a = input()

print(a)

x = np.array([[1,2,3], [4,5,6]])
i = 0
print("x.shape[%d] = %s" % (i, x.shape[i]))
print("x.shape = {}".format(x.shape))
