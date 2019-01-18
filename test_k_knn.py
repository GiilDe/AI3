from hw3_utils import *
from classifier import *


examples, labels, test_features = load_data()
data = []
data.append(examples)
data.append(labels)
split_crosscheck_groups(data, 2)
test_parameter_knn(range(1, 15, 2), 2)
