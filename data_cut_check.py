
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

for feature_num in range(5, 180, 8):
    print("data cut to: %d\n" % feature_num)
    data_new = []
    data_new.append(SelectKBest(f_classif, feature_num).fit_transform(examples, labels))
    data_new.append(labels)
    classifier.split_crosscheck_groups(data_new, 2)

    forest = classifier.sklearn_factory_wrapper(RandomForestClassifier())
    perceptron = classifier.sklearn_factory_wrapper(Perceptron())
    knn = classifier.knn_factory(7)

    print("forest perf:\n")
    accuracy, error = classifier.evaluate(forest, 2)
    print("%.3f, %.3f\n" % (accuracy, error))
    print("perc perf:\n")
    accuracy, error = classifier.evaluate(perceptron, 2)
    print("%.3f, %.3f\n" % (accuracy, error))
    print("knn perf:\n")
    accuracy, error = classifier.evaluate(knn, 2)
    print("%.3f, %.3f\n" % (accuracy, error))
    print("-----------------------------------\n")

