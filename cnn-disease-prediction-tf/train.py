import tensorflow as tf
import numpy as np
import processing_data
from disease_cnn import DiseaseCNN
import time
import os
import datetime

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
tf.flags.DEFINE_integer("batch_size", 8, "Batch Size (default: 8)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")


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
# x_windows, y = [[window1], [window2], ], [value1, value2]]
# window = [[day1], [day2], ...],    value : integer

# reshape y as [??, num_classes]
y_tmp = np.array(y)
y_tmp = np.reshape(y_tmp, (-1, 1))

# Randomly shuffle data
x_tmp = np.array(x_windows)
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
            window_height=x_train.shape[1],
            num_features=x_train.shape[2],
            num_classes= 1,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

    # window_height = the height of window that represent the number of days needed to predict
    # num_features = the number of features
    # num_classes = the number of label to predict
    # filter_size = size of filter(height)


        # Define Training procedure(compute gradient and apply gradient)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        # TODO : grads_and_vars check
        # train_op is train step

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        RMSE_summary = tf.scalar_summary("RMSE", cnn.RMSE)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, RMSE_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, RMSE_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        # define strain step session
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, RMSE = sess.run(
             [train_op, global_step, train_summary_op, cnn.loss, cnn.RMSE], feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, RMSE {:g}".format(time_str, step, loss, RMSE))
            # print data flow #

            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = { 
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0 
            }
            step, summaries, loss, RMSE = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.RMSE],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, RMSE {:g}".format(time_str, step, loss, RMSE))
            if writer:
                writer.add_summary(summaries, step)


        # Generate batches
        batches = processing_data.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch, one train step
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
