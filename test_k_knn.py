from hw3_utils import *
from classifier import *


examples, labels, test_features = load_data()
data = []
data.append(examples)
data.append(labels)
test_parameter_knn([i for i in range(1, 15, 2) if i not in {9, 11}], 2)
