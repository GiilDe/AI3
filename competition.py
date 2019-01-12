
import classifier
from hw3_utils import load_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier


examples, labels, test = load_data()
data = []
data.append(examples)
data.append(labels)
data_new = []
data_new.append(SelectKBest(f_classif, 100).fit_transform(examples, labels))
data_new.append(labels)
classifier.split_crosscheck_groups(data_new, 2)


print("using CUT data\n")

decision_tree = classifier.sklearn_factory_wrapper(RandomForestClassifier())
perceptron = classifier.sklearn_factory_wrapper(Perceptron())
knn = classifier.knn_factory(7)
print("knn and perceptron: \n")
ensemble = classifier.ensemble_factory([knn, perceptron])
accuracy, error = classifier.evaluate(ensemble, 2)
print("%.3f, %.3f\n" % (accuracy, error))

print("knn and decision tree: \n")
ensemble = classifier.ensemble_factory([knn, perceptron])
accuracy, error = classifier.evaluate(ensemble, 2)
print("%.3f, %.3f\n" % (accuracy, error))

print("all three: \n")
ensemble = classifier.ensemble_factory([knn, perceptron, decision_tree])
accuracy, error = classifier.evaluate(ensemble, 2)
print("%.3f, %.3f\n" % (accuracy, error))

print("knn alone: \n")
ensemble = classifier.ensemble_factory([knn])
accuracy, error = classifier.evaluate(ensemble, 2)
print("%.3f, %.3f\n" % (accuracy, error))


print("using UNCUT data\n")
classifier.split_crosscheck_groups(data, 2)

print("knn and perceptron: \n")
ensemble = classifier.ensemble_factory([knn, perceptron])
accuracy, error = classifier.evaluate(ensemble, 2)
print("%.3f, %.3f\n" % (accuracy, error))

print("knn and decision tree: \n")
ensemble = classifier.ensemble_factory([knn, perceptron])
accuracy, error = classifier.evaluate(ensemble, 2)
print("%.3f, %.3f\n" % (accuracy, error))

print("all three: \n")
ensemble = classifier.ensemble_factory([knn, perceptron, decision_tree])
accuracy, error = classifier.evaluate(ensemble, 2)
print("%.3f, %.3f\n" % (accuracy, error))

print("knn alone: \n")
ensemble = classifier.ensemble_factory([knn])
accuracy, error = classifier.evaluate(ensemble, 2)
print("%.3f, %.3f\n" % (accuracy, error))