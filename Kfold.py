import numpy as np
from sklearn.model_selection import KFold

def K_fold(dataset, label, split):
    train_index=[]
    test_index=[]
    kf = KFold(n_splits = split, shuffle=True)
    kf.split(dataset)
    for train_i, test_i in kf.split(dataset):
        train_index.append(train_i)
        test_index.append(test_i)

    return train_index, test_index