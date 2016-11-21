import tensorflow as tf
from pprint import pprint

from test_import import *

a_class = A()
#a_class._single_score()
#print("def(a_class) = %s" % dir(a_class))
#a_class._A__double_underscore()
#print(vars(a_class))

class B:
	def	__init__(self):
		self.value = 0


# Data loading params
tf.flags.DEFINE_string("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file_location", "./data/rt-polarity.pos", "Data source")

# Make flag, add flag data(._parse_flags())
FLAGS = tf.flags.FLAGS	#empty flag data structure
FLAGS._parse_flags()	#add flag data
print ("\nParameters:")

# Print flag value
for attr, value in sorted(FLAGS.__flags.items()):
	print("%s = %s" % (attr.upper(), value))
	#print("{} = {}".format(attr.upper(), value))
print("")

# Load data
print("Loading data...")

max_document_lenght = max([len(x.split(" ")) for x in x_text



