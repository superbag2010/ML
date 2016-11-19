import tensorflow as tf
import numpy as np

class DiseaseCNN(object):
	
	def __init__(self, num_attributes, num_classes, filter_sizes, num_filters, 12_reg_lambda=0.0):
		# Plcaeholders for input, output and dropout
		self.input_x = tf.placeholder(tf.float32, [None, num_attributes], name="input_x")
		self.inpuy_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# Keeping track of 12 regularization loss (optional)
		12_loss = tf.constant(0.0)


		# Embedding layer
		with tf.device('\cpu:0'), tf.name_scope("embedding"):
			W = random_uniform([
