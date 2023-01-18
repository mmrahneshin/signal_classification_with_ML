import numpy as np

from select_features.featureAccuracy import feature_accuracy

def select_feature(x, y, train_index, test_index):
    feature_score = feature_accuracy(x, y, train_index, test_index)
    