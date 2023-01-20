from select_features.main import select_feature
from Kfold import K_fold

from clustering.main import clustering

def CD_E(x,y):

    x = x[200:500]
    y = y[200:500]
    
    train_index, test_index = K_fold(x, 5)
    selected_features = select_feature(x,y, train_index, test_index, 5)
    
    clustering(selected_features,y,10)