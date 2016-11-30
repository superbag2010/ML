import tensorflow as tf
import numpy as np

x = np.array([[0,1,2], [3,4,5], [6,7,8], [9,10,11]])

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(4))


print("x = \n{}".format(x))

print("shuffle_indices = {}".format(shuffle_indices))

x_shuffled = x[shuffle_indices]
print("x_shuffled = \n{}".format(x_shuffled))
