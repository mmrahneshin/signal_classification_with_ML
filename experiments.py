import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.svm import SVC
from decimal import *

from features.initial import initialize_features
from Kfold import K_fold

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

    x_features = initialize_features(x)

    print(x_features.shape)

    train_index, test_index = K_fold(x_features, y, 5)

    for train_i, test_i in zip(train_index,test_index):

        x_train = x_features[train_i]
        x_test = x_features[test_i]
        y_train = y[train_i]
        y_test = y[test_i]

        print(x_test.shape)

        clf = SVC(kernel='linear')
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        print(Decimal(accuracy_score(y_test,y_pred)))

   

   

main()