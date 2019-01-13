from id3 import Id3Estimator
from hw3_utils import load_data
from id3 import export_graphviz
import classifier

examples, labels, test = load_data()
data = list()
data.append(examples)
data.append(labels)

examples_len = len(examples)

training_set_examples = list(examples[0:int(examples_len*(8/10))])
training_set_examples.extend(examples[int(examples_len*(9/10)):])
training_set_labels = list(labels[0:int(examples_len*(8/10))])
training_set_labels.extend(labels[int(examples_len*(9/10)):])

test_set_examples = examples[int(examples_len*(8/10)):int(examples_len*(9/10))]
test_set_labels = labels[int(examples_len*(8/10)):int(examples_len*(9/10))]

classifier.split_crosscheck_groups(data, 2)

estimator = Id3Estimator()
estimator.fit(training_set_examples, training_set_labels)

predicted_labels = estimator.predict(test_set_examples)

correct = 0
for i in range(len(test_set_labels)):
    if predicted_labels[i] == test_set_labels[i]:
        correct += 1


export_graphviz(estimator.tree_, 'tree.dot')
accuracy = correct/(len(test_set_labels))

print(accuracy)
