import tensorflow as tf
import processing_data
import numpy as np

x_attrs, y = processing_data.load_data_and_labels("./data/data.csv")

#def __init__(self, window_height, num_attr, num_classes, filter_sizes, num_filters, 12_reg_lambda=0.0):

x_attrs = np.array(x_attrs)
num_attrs = x_attrs.shape[1]
num_days = x_attrs.shape[0]
window_height = 3

x_windows = list()
y = y[window_height-1:]

for i in range(num_days):
    index_end = i + window_height
    x_windows = x_windows + [x_attrs[i:index_end]]
    if (index_end >= num_days):
        break

k = -1
print("x[{}] = {}, y[{}] = {}".format(k, x_windows[k], k, y[k]))


y = np.array(y)
num_classes = 15

# Plcaeholders for input, output and dropout
input_x = tf.placeholder(tf.float32, [None, num_attrs], name="input_x")
inpuy_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# Keeping track of 12 regularization loss (optional)
#12_loss = tf.constant(0.0)

# Grouping layer(Grouping input_x by window_height)
with tf.device('/cpu:0'), tf.name_scope("Grouping_days"):
    print()

