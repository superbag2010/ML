import tensorflow as tf
import numpy as np

x = np.array([[0,1,2], [3,4,5], [6,7,8], [9,10,11]])

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(4))


print("x = \n{}".format(x))

print("shuffle_indices = {}".format(shuffle_indices))

x_shuffled = x[shuffle_indices]
print("x_shuffled = \n{}".format(x_shuffled))


x = tf.constant(np.array([[0.0,1.0,2.0], [3.0,4.0,5.0], [0.6,0.7,0.8], [9.0,10.0,11.0]]))
x_soft = tf.nn.softmax(x, -1)
sess = tf.Session()
print(sess.run(x_soft))


a = [1,2]
a.append([3])
print(a)
