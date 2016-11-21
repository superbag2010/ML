import re
import numpy as np

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


positive_examples = list(open("/root/cnn-text-classification-tf-master/data/rt-polaritydata/rt-polarity.pos", "r").readlines())
positive_examples = [s.strip() for s in positive_examples]
negative_examples = list(open("/root/cnn-text-classification-tf-master/data/rt-polaritydata/rt-polarity.neg", "r").readlines())
negative_examples = [s.strip() for s in negative_examples]

# Split by words
x_text = positive_examples + negative_examples
x_text = [clean_str(sent) for sent in x_text]

# Generate labels
#positive_labels = [1 for _ in positive_examples]


print (x_text[0])
print (x_text[1])


print ("AB".lower())

[a, b] = [[[3,3,3], [x,x]] for x in range(2)]
c = [a, b]
print ("a = {}".format(a))
print ("b = {}".format(b))
print ("c[1] = {}".format(c[1]))

