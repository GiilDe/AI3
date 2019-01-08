
from hw3_utils import abstract_classifier, load_data
import numpy as np


def euclidean_distance(feature_list1, feature_list2):
    dist = 0
    for feature1, feature2 in zip(feature_list1, feature_list2):
        dist += (feature1-feature2)**2
    return np.sqrt(dist)


class knn_classifier(abstract_classifier):
    def __init__(self, training_set, training_labels, k):
        self.K = k
        self.training_set = training_set
        self.training_labels = training_labels

    def classify(self, features):
        examples_distances = [euclidean_distance(features, example) for example in self.training_set]
        kth_distance = np.partition(examples_distances, self.K)[self.K]
        kth_smallest = [1 if label is True else -1 for (example, label) in zip(self.training_set, self.training_labels)
                        if euclidean_distance(features, example) <= kth_distance]
        return sum(kth_smallest) > 0
