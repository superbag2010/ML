import tensorflow as tf
import processing_data
import numpy as np

x_attr, y = processing_data.load_data_and_labels("./data/data.csv")

#def __init__(self, stride_window_size, num_attr, num_classes, filter_sizes, num_filters, 12_reg_lambda=0.0):

print("x_attr = {}".format(x_attr))
x_attr = np.array(x_attr)
num_attr = x_attr.shape[1]
print("x_attr[3] = {}".format(x_attr[3]))
#print("x_attr = {}".format(x_attr))

y = np.array(y)
num_classes = 15

# Plcaeholders for input, output and dropout
input_x = tf.placeholder(tf.float32, [None, num_attr], name="input_x")
inpuy_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# Keeping track of 12 regularization loss (optional)
#12_loss = tf.constant(0.0)

# Grouping layer(Grouping input_x by stride_window_size)
with tf.device('/cpu:0'), tf.name_scope("Grouping_days"):
    print()
