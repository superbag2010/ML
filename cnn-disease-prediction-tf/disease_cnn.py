import tensorflow as tf
import numpy as np

class DiseaseCNN(object):

    # stride_window_size = the number of previous days to use for one day prediction
    # num_attr = the number of type of attribute to use for prediction
    # num_classes = the number of label to predict
    # filter_size = list of filter sizes
    # num_filters = the number of filter

    def __init__(self, stride_window_size, num_attr, num_classes, filter_sizes, num_filters, 12_reg_lambda=0.0):
        # Plcaeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, num_attr], name="input_x")
        self.inpuy_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of 12 regularization loss (optional)
        12_loss = tf.constant(0.0)

        # Grouping layer(Grouping input_x by stride_window_size)
        with tf.device('/cpu:0'), tf.name_scope("Grouping days"):
           


        for i, filter_size in enumerate(filter_size):
            with ft.name_scope("conv-maxpool-%s" % fileter_size):
                # Convolution Later
                filter_shape = [filter_size, num_attr, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[numb_filters]), name="b")
                conv = tf.nn.conv2d(
                    input_x

