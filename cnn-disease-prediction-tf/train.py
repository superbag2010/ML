import tensorflow as tf
import processing_data

# ============================================================
# Set flags
# ============================================================
# Data loading params
tf.flags.DEFINE_string("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file_location", "./data/data.csv", "Data source")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


# Make flag, add flag data(._parse_flags())
FLAGS = tf.flags.FLAGS  #empty flag data structure
FLAGS._parse_flags()    #add flag data
print ("==========================================\nParameters:")

# Print flag value
for attr, value in sorted(FLAGS.__flags.items()):
    print("%s = %s" % (attr.upper(), value))
    #print("{} = {}".format(attr.upper(), value))		
print ("==========================================\n")


# ============================================================
# Data Preparation
# ============================================================
# Load data
print("Loading data...")
x_features, y = processing_data.load_data_and_labels(FLAGS.data_file_location)

# SPlit train/test set
# TODO: this is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_trian, x_dev = x_features[:dev_sample_index], x_features[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
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
      # with sess.as_default():




