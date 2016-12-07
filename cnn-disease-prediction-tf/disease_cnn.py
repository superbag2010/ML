#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import processing_data

class DiseaseCNN(object):

    # window_height = the number of previous days to use for one day prediction
    # num_features = the number of type of attribute to use for prediction
    # num_nodes = the number of each layer's node
    # num_classes = the number of label to predict
    # filter_size = list of filter sizes
    # num_filters = the number of filter

    def __init__(self, window_height, num_features, num_nodes ,num_classes, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Plcaeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, window_height, num_features], name="input_x")
        self.expanded_input_x = tf.expand_dims(self.input_x, -1)
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of 12 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Later
                filter_shape = [filter_size, num_features, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.expanded_input_x,
                    W,                  # filter
                    strides=[1,1,1,1],
                    padding="VALID",    # no padding
                    name="conv")
                # shape(self.x_window) : [days, window_height, num_features, 1]
                # days : the number of days to predict
                # window_height : the number of days of one window set

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # shape(h) : [days, window_height - filter_size + 1, 1, num_filters]

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, window_height - filter_size + 1, 1, 1],
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="pool")
                pooled_outputs.append(pooled)
                # shape(pooled) : [days, 1, 1, num_filters]
                # shape(pooled_outputs)
                # [[days, 1, 1, num_filters], [days, 1, 1, num_filters], [days, 1, 1, num_filters]]

        
        # Combine all the pooled featrues
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        # shape(h_pool) : [days, 1, 1, num_filters * the number of filter size]

        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # shape(h_pool_flat) : [days, num_filters * the number of filter size]

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        # if dropout_keep_prob = 1.0, can use as eval model

        pre_num_node = num_filters_total
        self.NN_result = [None] * (len(num_nodes) + 1)
        self.NN_result[0] = self.h_drop
        # NN layer
        for i, num_node in enumerate(num_nodes):
            with tf.name_scope("Completely_connected_NN{}".format(i)):
                num_nodes = num_node
                W = tf.get_variable(
                    "W{}".format(i),
                    shape = [pre_num_node, num_node],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_node]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.NN_result[i+1] = tf.nn.xw_plus_b(self.NN_result[i], W, b, name="NN_result{}".format(i+1))
                pre_num_node = num_node

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape = [pre_num_node, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # tf.nn.l2_loss(a) = sum(a^2)/2, element-wise
            self.scores = tf.nn.xw_plus_b(self.NN_result[-1], W, b, name="scores")
            # scores = XW + b
            # scores = [value1 about window1, value2, ....]

        # CalculateMean cross-entropy loss(as l2_reg_lambda, L2 reg is applied)
        with tf.name_scope("loss"):
            losses = tf.square(tf.sub(self.scores, self.input_y))
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        # L2 reg => lambda*(w^2)/2
        # mean MSE

        # Accuracy
        with tf.name_scope("RMSE"):
            subtraction = tf.sub(self.scores, self.input_y)
            deviation = tf.square(subtraction)
            MSE = tf.sqrt(deviation)
            self.RMSE = tf.reduce_mean(MSE, name="RMSE")
