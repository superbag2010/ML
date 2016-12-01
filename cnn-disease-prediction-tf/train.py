import tensorflow as tf
import numpy as np
import processing_data
from disease_cnn import DiseaseCNN

# ============================================================
# Set flags
# ============================================================
# Data loading params
tf.flags.DEFINE_string("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file_location", "./data/data.csv", "Data source")

# Model Hyperparameters
tf.flags.DEFINE_integer("num_features", 21, "the number of feature attributes")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Model Hyperparameters
tf.flags.DEFINE_integer("window_height", 14, "the number of days needed to predict one day")


# Make flag, add flag data(._parse_flags())
FLAGS = tf.flags.FLAGS  # empty flag data structure
FLAGS._parse_flags()    # add flag data
print ("==========================================\nParameters:")

# Print flag value
for attr, value in sorted(FLAGS.__flags.items()):
    print("%s = %s" % (attr.upper(), value))
    #print("{} = {}".format(attr.upper(), value))		
print ("==========================================\n")


# ============================================================
# Data Preparation
# ============================================================
print("Loading data...")
# Load data
x_features, y_input = processing_data.load_data_and_labels(FLAGS.data_file_location)
# x_features, y = [[[Feature1(day)1, F12, ...], [F21, F22, ...], ...], [Label1, L2, ...] ]

# Grouping data
x_windows, y = processing_data.grouping_data(x_features, y_input, FLAGS.window_height)
# x_features, y = [[window1, window2, ], [value1, value2]]
# window = [day1, day2, ...],    value : integer

# Randomly shuffle data
x_tmp = np.array(x_windows)
y_tmp = np.array(y)
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x_tmp[shuffle_indices]
y_shuffled = y_tmp[shuffle_indices]


# SPlit train/test set
# TODO: this is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: %s/%s" % (len(y_train), len(y_dev)))


# ============================================================
# Training
# ============================================================
with tf.Graph().as_default():
# TODO: What is soft_placement, log_device_placement
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = DiseaseCNN(
            window_height=x_train.shape[0],
            num_features=x_train.shape[1],
            num_classes= 1,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

    # window_height = the height of window that represent the number of days needed to predict
    # num_features = the number of features
    # num_classes = the number of label to predict
    # filter_size = size of filter(height)


    #Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)

