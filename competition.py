from classifier import *
from hw3_utils import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier


examples, labels, test = load_data()
data = []
data.append(examples)
data.append(labels)
split_crosscheck_groups(data, 2)

forest = sklearn_factory_wrapper(RandomForestClassifier())
perceptron = sklearn_factory_wrapper(Perceptron())
knn3 = knn_factory(3)
knn7 = knn_factory(7)
knn11 = knn_factory(11)

print("knn7: \n")
ensemble = ensemble_factory([knn7])
accuracy, error = evaluate(ensemble, 2)
print("%.3f, %.3f\n" % (accuracy, error))

print("knn3,7,11: \n")
ensemble = ensemble_factory([knn3, knn7, knn11])
accuracy, error = evaluate(ensemble, 2)
print("%.3f, %.3f\n" % (accuracy, error))

