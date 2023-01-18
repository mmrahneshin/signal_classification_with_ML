import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter

from sklearn.preprocessing import StandardScaler
from features.initial import initialize_features
from classification.SVM import SVM
from classification.KNN import KNN
from classification.randomForest import random_forest
from Kfold import K_fold
from chart.confusionMatrix import confMatrix
from chart.rocMatrix import roc_matrix
from select_features.main import select_feature

import random
import os
seed = 57

def main():
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


    x = pickle.load(open('x.pkl', 'rb'))
    y = pickle.load(open('y.pkl', 'rb'))

    x_normal = np.concatenate((x[:300], x[400:]), axis=0)
    x_seizure = x[300:400]
    print(x_normal.shape)
    print(x_seizure.shape)
    sampling_freq = 173.6 #based on info from website

    b, a = butter(3, [0.5,40], btype='bandpass',fs=sampling_freq)


    x_normal_filtered = np.array([lfilter(b,a,x_normal[ind,:]) for ind in range(x_normal.shape[0])])
    x_seizure_filtered = np.array([lfilter(b,a,x_seizure[ind,:]) for ind in range(x_seizure.shape[0])])
    print(x_normal.shape)
    print(x_seizure.shape)


    x_normal = x_normal_filtered
    x_seizure = x_seizure_filtered

    x = np.concatenate((x_normal,x_seizure))
    y = np.concatenate((np.zeros((400,1)),np.ones((100,1))))
    
    print(x.shape)
    print(y.shape)
    # standard--------------------------------------------
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    # standard--------------------------------------------

  
    x_features = initialize_features(x)

  
    print(x_features.shape)

    train_index, test_index = K_fold(x_features, 5)

    # phase 2 -----------------------------------------------
    select_feature(x_features, y, train_index, test_index)
    # phase 2 -----------------------------------------------

    clf_score_save = []

    # for train_i, test_i in zip(train_index,test_index):

    #     x_train = x_features[train_i]
    #     x_test = x_features[test_i]
    #     y_train = y[train_i]
    #     y_test = y[test_i]

    #     # acc_score, clf = SVM(x_train, y_train, x_test, y_test)
    #     # clf_score_save.append([acc_score, clf, x_test, y_test])

    #     # acc_score, clf = KNN(x_train, y_train, x_test, y_test)
    #     # clf_score_save.append([acc_score, clf, x_test, y_test])

    #     acc_score, clf = random_forest(x_train, y_train, x_test, y_test)
    #     clf_score_save.append([acc_score, clf, x_test, y_test])

    # acc_score, clf, x_test, y_test = find_best(clf_score_save)
    # confMatrix(x_test, y_test, clf)
    # roc_matrix(x_test, y_test, clf)


def find_best(data):
    best = None
    max = -1
    for node in data:
        if node[0] > max:
            best = node
            max = node[0]

    return best

main()