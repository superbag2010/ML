import tensorflow as tf
import numpy as np


vocab_size = tf.Variable(3, name="Vsize")
embedding_size = tf.Variable(1, name="Esize")

init = tf.initialize_all_variables()

with tf.Session() as sess:
	with tf.device('/cpu:0'), tf.name_scope("embedding"):
		w = tf.random_uniform([1, 3], -1.0, 1.0)
		sess.run(init)
		result = sess.run(w)
		print(result)



#	self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
#	self.embedded_chars_expanded = tf.expand_dims(self.embeeded_chars, -1)



#embbedded_chars = tf.nn.embedding_lookup(

