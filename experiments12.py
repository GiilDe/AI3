from sklearn import tree
from sklearn.linear_model import Perceptron
import classifier, hw3_utils


# experiment 1
examples, labels, test = hw3_utils.load_data()
data = []
data.append(examples)
data.append(labels)
classifier.split_crosscheck_groups(data, 2)

tree_classifier = tree.DecisionTreeClassifier(criterion='entropy')
accuracy, error = classifier.evaluate(tree_classifier, 2)
print("%.2f, %.2f" % (accuracy, error))

# experiment 2

perceptron = Perceptron()
accuracy, error = classifier.evaluate(perceptron, 2)
print("%.2f, %.2f" % (accuracy, error))