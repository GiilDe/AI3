
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
        return sum(bool_to_int(label) for (example, label) in zip(self.training_set, self.training_labels)
                   if euclidean_distance(features, example) <= kth_distance) > 0


class ensemble_factory(abstract_classifier_factory):
    def __init__(self, factories):
        self.factories = factories

    def train(self, data, labels):
        classifiers = [factory.train(data, labels) for factory in self.factories]
        return classifier_ensemble(classifiers)


class classifier_ensemble(abstract_classifier):
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def classify(self, features):
        return sum(bool_to_int(classifier.classify(features)) for classifier in self.classifiers) > 0


class sklearn_factory_wrapper(abstract_classifier_factory):
    def __init__(self, classifier_factory):
        self.classifier_factory = classifier_factory

    def train(self, data, labels)->abstract_classifier:
        return sklearn_classifier_wrapper(self.classifier_factory.fit(data, labels))


class sklearn_classifier_wrapper(abstract_classifier):
    def __init__(self, classifier):
        self.classifier = classifier

    def classify(self, features):
        shaped = np.array(features).reshape(1, len(features))
        return self.classifier.predict(shaped)


def bool_to_int(b: bool):
    res = 1 if b else -1
    return res


def euclidean_distance(feature_list1, feature_list2):
    s = sum((feature1-feature2)**2 for feature1, feature2 in zip(feature_list1, feature_list2))
    return np.sqrt(s)


def get_accuracy(classifier: abstract_classifier, test_set, test_labels):
    positive = 0
    negative = 0
    for (features, label) in zip(test_set, test_labels):
        if classifier.classify(features) == label:
            positive = positive + 1
        else:
            negative = negative + 1
    return positive/len(test_set), negative/len(test_set)


def evaluate(classifier_factory: abstract_classifier_factory, k):
    all_examples = []
    all_labels = []
    for i in range(0, k):
        examples, labels = load_k_fold_data(i)
        all_examples.append(examples)
        all_labels.append(labels)
    return k_fold_cross_validation(classifier_factory, all_examples, all_labels, k)


# assuming that len(data) is divisible by given k
def k_fold_cross_validation(classifier_factory: abstract_classifier_factory, data, labels, k):
    accuracies = []
    errors = []
    for i in range(0, k):
        test_set = data[i]
        test_labels = labels[i]
        r = 0 if i != 0 else 1
        current_training = data[r]
        current_labels = labels[r]
        for j in range(r + 1, k):
            if j != i:
                current_training = np.concatenate((current_training, data[j]))
                current_labels = np.concatenate((current_labels, labels[j]))
        classifier = classifier_factory.train(current_training, current_labels)
        accuracy, error = get_accuracy(classifier, test_set, test_labels)
        accuracies.append(accuracy)
        errors.append(error)
    return np.average(accuracies), np.average(errors)


def test_parameter_knn(possible_values, k):
    results = [(value, evaluate(knn_factory(value), k)) for value in possible_values]
    accuracies = [accuracy for (value, (accuracy, error)) in results]
    errors = [error for (value, (accuracy, error)) in results]
    print(results)
    return np.average(accuracies), np.average(errors)

def split_crosscheck_groups(dataset, num_folds):
    examples = list(dataset[0])
    labels = dataset[1]
    true_rate = labels.count(True) / len(labels)
    examples_in_group = len(dataset[0]) / num_folds
    true_in_group = int(examples_in_group * true_rate)
    false_in_group = examples_in_group - true_in_group

    for i in range(1, num_folds):
        current_examples = []
        current_labels = []
        current_true = 0
        current_false = 0
        current_size = 0
        index = 0
        while current_size < examples_in_group:
            if labels[index] == True and current_true < true_in_group:
                current_labels.append(True)
                labels.pop(index)
                current_examples.append(examples[index])
                examples.pop(index)
                current_size += 1
            elif labels[index] == False and current_false < false_in_group:
                current_labels.append(False)
                labels.pop(index)
                current_examples.append(examples[index])
                examples.pop(index)
                current_size += 1
            else:
                index += 1
        pickle.dump((current_examples, current_labels), open('ecg_fold_%d' % i, 'wb'))
    pickle.dump((examples, labels), open('ecg_fold_%d' % num_folds, 'wb'))


# dataset = zip(examples, labels), assumes len(data) is divisible by num_folds
def old_split_crosscheck_groups(dataset, num_folds):
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
