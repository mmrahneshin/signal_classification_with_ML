import numpy as np

from classification.ID3 import ID3


def feature_accuracy(x, y, train_index, test_index):
    x_feature = x.T
    feature_score = []
    
    for feature in x_feature:
        acc_score = []
        for train_i, test_i in zip(train_index,test_index):
            x_train = feature[train_i].reshape(-1,1)
            x_test = feature[test_i].reshape(-1,1)
            y_train = y[train_i]
            y_test = y[test_i]
            acc_score.append(ID3(x_train, y_train, x_test, y_test))
        feature_score.append(np.mean(acc_score))

    feature_score = np.array(feature_score)

    return feature_score