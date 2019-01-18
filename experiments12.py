from classifier import *
from hw3_utils import load_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron


# experiment 1
examples, labels, test = load_data()
data = list()
data.append(examples)
data.append(labels)
split_crosscheck_groups(data, 2)

tree_classifier = sklearn_factory_wrapper(DecisionTreeClassifier(criterion='entropy'))
accuracy, error = evaluate(tree_classifier, 2)
print("%.2f, %.2f" % (accuracy, error))

# experiment 2
perceptron = sklearn_factory_wrapper(Perceptron())
accuracy, error = evaluate(perceptron, 2)
print("%.2f, %.2f" % (accuracy, error))

