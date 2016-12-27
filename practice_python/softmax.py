import tensorflow as tf
import numpy as np

with tf.Session() as sess:
  a = tf.Variable(tf.constant([5.4, 1.3]))
  softmax = tf.nn.softmax(a)
  sess.run(tf.initialize_all_variables())
  print(sess.run(softmax))

