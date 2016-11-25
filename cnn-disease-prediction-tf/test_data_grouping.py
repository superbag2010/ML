import tensorflow as tf
import processing_data
import numpy as np

x_attr, y = processing_data.load_data_and_labels("./data/data.csv")

#def __init__(self, window_height, num_attr, num_classes, filter_sizes, num_filters, 12_reg_lambda=0.0):

x_attr = np.array(x_attr)
num_attr = x_attr.shape[1]
num_day = x_attr.shape[0]
window_height = 1

x = list()

for i in range(num_day):
    index_end = i + window_height
    x = x + [x_attr[i:index_end]]
    if (index_end >= num_day):
        break

print("x[1] = {}".format(x[1]))


y = np.array(y)
num_classes = 15

# Plcaeholders for input, output and dropout
input_x = tf.placeholder(tf.float32, [None, num_attr], name="input_x")
inpuy_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# Keeping track of 12 regularization loss (optional)
#12_loss = tf.constant(0.0)

# Grouping layer(Grouping input_x by window_height)
with tf.device('/cpu:0'), tf.name_scope("Grouping_days"):
    print()

