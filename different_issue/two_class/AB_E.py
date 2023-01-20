import numpy as np
from select_features.main import select_feature
from Kfold import K_fold

from clustering.main import clustering

def AB_E(x,y):

    x = np.concatenate((x[:200], x[400:500]))
    y = np.concatenate((y[:200], y[400:500]))
     
    train_index, test_index = K_fold(x, 5)
    selected_features = select_feature(x,y, train_index, test_index, 10)
    
    clustering(selected_features,y,15)