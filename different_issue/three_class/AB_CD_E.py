from select_features.main import select_feature
from Kfold import K_fold

from clustering.main import clustering

def AB_CD_E(x,y):
    
    y[:200] = 1
    y[400:] = 0

    train_index, test_index = K_fold(x, 5)
    selected_features = select_feature(x,y, train_index, test_index, 10)
    
    clustering(selected_features,y, 10)