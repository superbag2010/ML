#! /usr/bin/python3

import tensorflow as tf
import numpy as np

state = tf.Variable(0)
one = tf.constant(5)
input_x = tf.placeholder(tf.int32)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)
result = tf.add(update, input_x)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
#  print("new_value = {}".format(sess.run(new_value)))
#  print("update = {}".format(sess.run(update)))
  print("result = {}".format(sess.run(result, feed_dict={input_x : 5})))
