import re


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
    y = list()

    # Find the number of attribute(featurei + label), suppose all tuple have the same number of attribute.
    feature_len = len(clean_str(disease_examples[0]).split(",")) - 1

    for one_day in disease_examples:
        one_day = clean_str(one_day)
       	attributes = list(map(float, one_day.split(",")))	#change to float
		
       	feature = list(attributes[:feature_len])
       	x_features.append(feature)
       	y = y + [attributes[-1]]

        return [x_features, y]



if __name__ == "__main__":
	"""
	print data values are well loaded(test this file).
	"""
	l = load_data_and_labels("./data/data.csv")
	print (l[0])
	print (l[1][0])
