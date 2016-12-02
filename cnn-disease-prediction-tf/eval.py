import tensorflow as tf
import numpy as np
import os
import time
import datetime
import processing_data
from disease_cnn import DiseaseCNN
import csv

# Parameters
# =========================================

# Data Parameters
tf.flags.DEFINE_string("data_file_location", "./data/data.csv", "Data source")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Model Hyperparameters
tf.flags.DEFINE_integer("window_height", 14, "the number of days needed to predict one day")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    #Load data from file
    x_features, y_input = processing_data.load_data_and_labels(FLAGS.data_file_location)
    # x_features, y = [[[Feature1(day)1, F12, ...], [F21, F22, ...], ...], [Label1, L2, ...] ]

    # Grouping data
    x_eval, y = processing_data.grouping_data(x_features, y_input, FLAGS.window_height)
    # x_eval, y = [[window1], [window2], ], [value1, value2]]
    # window = [[day1], [day2], ...],    value : integer

    # reshape y as [??, num_classes]
    y_tmp = np.array(y)
    y_eval = np.reshape(y_tmp, (-1, 1))

else:
    x_raw = []
    y_test = []


# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        # Generate batches for one epoch
        batches = processing_data.batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_scores = []

        for x_test_batch in batches:
            batch_scores = sess.run(scores, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            for i in batch_scores:
                all_scores.append(int(i))




# Print accuracy if y_test is defined
if y_eval is not None:
    print("Total number of test examples: {}".format(len(y_eval)))
    print("all_scores : \n{}".format(all_scores))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((all_scores, y_eval))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
