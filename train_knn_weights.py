from hw3_utils import *
import numpy as np
from random_hill_climbing import hill_climbing
from classifier import load_k_fold_data, split_crosscheck_groups, k_fold_cross_validation


def weighted_euclidean_distance(feature_list1, feature_list2, weights):
    s = sum([(weight*(feature1-feature2)**2) for feature1, feature2, weight in zip(feature_list1, feature_list2, weights)])
    return np.sqrt(s)


def bool_to_int(b: bool):
    res = 1 if b else -1
    return res


class knn_factory(abstract_classifier_factory):
    def __init__(self, k, weights):
        self.K = k
        self.weights = weights

    def train(self, data, labels):
        return weighted_knn_classifier(data, labels, self.K, self.weights)


class weighted_knn_classifier(abstract_classifier):
    def __init__(self, training_set, training_labels, k, weights):
        self.K = k
        self.training_set = training_set
        self.training_labels = training_labels
        self.weights = weights

    def classify(self, features):
        examples_distances = [weighted_euclidean_distance(features, example, self.weights) for example in self.training_set]
        kth_distance = np.partition(examples_distances, self.K)[self.K]
        return sum(bool_to_int(label) for (example, label) in zip(self.training_set, self.training_labels)
                   if weighted_euclidean_distance(features, example, self.weights) <= kth_distance) > 0


def evaluate(classifier_factory: abstract_classifier_factory, k):
    all_examples = []
    all_labels = []
    for i in range(0, k):
        examples, labels = load_k_fold_data(i)
        all_examples.append(examples)
        all_labels.append(labels)
    return k_fold_cross_validation(classifier_factory, all_examples, all_labels, k)


def weights_score(weights):
    score = evaluate(knn_factory(5, weights), k)[0]
    print(score)
    return score


def train_weights(splits_num, train_len=None):
    examples, labels, test_features = load_data()
    if train_len != None:
        examples = examples[0:train_len]
        labels = labels[0:train_len]
    data = list()
    data.append(examples)
    data.append(labels)
    weights = np.zeros(len(examples[0]))
    split_crosscheck_groups(data, splits_num)
    weights = hill_climbing(f=weights_score, num_iter=100, directions=5, initial_step_size=2, N=5, best_N=None,
                            length=len(weights), T=5)
    return weights


k = 2
print(train_weights(k, 500))