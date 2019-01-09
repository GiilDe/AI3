
from hw3_utils import abstract_classifier, abstract_classifier_factory
import numpy as np
import pickle


class knn_factory(abstract_classifier_factory):
    def __init__(self, k):
        self.K = k

    def train(self, data, labels):
        return knn_classifier(data, labels, self.K)


class knn_classifier(abstract_classifier):
    def __init__(self, training_set, training_labels, k):
        self.K = k
        self.training_set = training_set
        self.training_labels = training_labels

    def classify(self, features):
        examples_distances = [euclidean_distance(features, example) for example in self.training_set]
        kth_distance = np.partition(examples_distances, self.K)[self.K]
        kth_smallest = [1 if label else -1 for (example, label) in zip(self.training_set, self.training_labels)
                        if euclidean_distance(features, example) <= kth_distance]
        return sum(kth_smallest) > 0


def euclidean_distance(feature_list1, feature_list2):
    dist = 0
    for feature1, feature2 in zip(feature_list1, feature_list2):
        dist += (feature1-feature2)**2
    return np.sqrt(dist)


def get_accuracy(classifier, test_set, test_labels):
    positive = 0
    negative = 0
    for (features, label) in zip(test_set, test_labels):
        shaped = np.array(features).reshape(1, 187)
        if classifier.predict(shaped) == label:
            positive = positive + 1
        else:
            negative = negative + 1
    return positive/len(test_set), negative/len(test_set)


def evaluate(classifier_factory, k):
    all_examples = []
    all_labels = []
    for i in range(0, k):
        examples, labels = load_k_fold_data(i)
        all_examples.append(examples)
        all_labels.append(labels)
    return k_fold_cross_validation(classifier_factory, all_examples, all_labels, k)


# assuming that len(data) is divisible by given k
def k_fold_cross_validation(classifier_factory, data, labels, k):
    accuracies = []
    errors = []
    for i in range(0, k):
        test_set = data[i]
        test_labels = labels[i]
        r = 0 if i != 0 else 1
        current_training = data[r]
        current_labels = labels[r]
        for j in range(r+1, k):
            if j != i:
                current_training = np.concatenate((current_training, data[j]))
                current_labels = np.concatenate((current_labels, labels[j]))
        classifier = classifier_factory.fit(current_training, current_labels)
        accuracy, error = get_accuracy(classifier, test_set, test_labels)
        accuracies.append(accuracy)
        errors.append(error)
    return np.average(accuracies), np.average(errors)


def test_parameter_knn(possible_values, k):
    results = [(value, evaluate(knn_factory(value), k)) for value in possible_values]
    accuracies = [accuracy for (value, accuracy, error) in results]
    errors = [error for (value, accuracy, error) in results]
    return np.average(accuracies), np.average(errors)


# dataset = zip(examples, labels), assumes len(data) is divisible by num_folds
def split_crosscheck_groups(dataset, num_folds):
    examples = dataset[0]
    labels = dataset[1]
    step = int(len(dataset[0])/num_folds)
    j = 0
    for i in range(0, len(dataset[0]), step):
        current_examples = examples[i:i+step]
        current_labels = labels[i:i+step]
        with open('ecg_fold_%d' % j, 'wb') as file:
            pickle.dump((current_examples, current_labels), file)
        j = j + 1


def load_k_fold_data(i):
    with open('ecg_fold_%d' % i, 'rb') as file:
        return pickle.load(file)