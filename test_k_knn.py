
import classifier, hw3_utils

examples, labels, test_features = hw3_utils.load_data()
data = []
data.append(examples)
data.append(labels)
classifier.split_crosscheck_groups(data, 2)
classifier.test_parameter_knn(range(1, 15, 2), 2)
