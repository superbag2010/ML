import re
import numpy as np

def clean_str(string):
    """
    remove space, "\n"
    """
    string = re.sub(r"\n", "", string)
    string = re.sub(r" ", "", string)
    return string


def load_data_and_labels(disease_data_file):
    """
    load data file, and [[f1 set, f2 set, ...], [l1, l2, ...]] format
    return [ [[features1-1, features1-2, features1-3], [f2-1, f2-2, f2-3], ....], [labe1, l2, ...]]
    """
    # Load data from files
    disease_examples = list(open(disease_data_file, "r").readlines())

    x_features = list()
    y_input = list()

    # Find the number of attribute(featurei + label), suppose all tuple have the same number of attribute.
    feature_len = len(clean_str(disease_examples[0]).split(",")) - 1

    for one_day in disease_examples:
        one_day = clean_str(one_day)
       	attributes = list(map(float, one_day.split(",")))	#change to float
       	feature = list(attributes[:feature_len])
       	x_features.append(feature)
       	y_input.append(attributes[-1])

    return [x_features, y_input]
    # return [ [[Feature11, F12, ...], [F21, F22, ...], ...], [L1, L2, ...] ]
    # F = float, L = float

def grouping_data(x_features, y, window_height):
    x_features = np.array(x_features)
    num_days = x_features.shape[0]

    x_windows = list()
    y = y[window_height-1:]

    for i in range(num_days):
        index_end = i + window_height
        x_windows = x_windows + [x_features[i:index_end]]
        if (index_end >= num_days):
            break
    return [x_windows, y]

if __name__ == "__main__":
    """
    print data values are well loaded(test this file).
    """
    x, y = load_data_and_labels("./data/data.csv")
    print (x[3])
#   print (l[0])
    
    x_windows, y = grouping_data(x, y, 3)
    k = -1
    print("x[{}] = {}, y[{}] = {}".format(k, x_windows[k], k, y[k]))
