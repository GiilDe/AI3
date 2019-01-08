
import numpy as np


def euclidean_distance(feature_list1, feature_list2):
    dist = 0
    for feature1, feature2 in zip(feature_list1, feature_list2):
        dist += (feature1-feature2)**2
    return np.sqrt(dist)

print(euclidean_distance((3,8),(5,7)))
