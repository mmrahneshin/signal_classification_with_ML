from select_features.main import select_feature
from Kfold import K_fold

from clustering.main import clustering


def AB_CD(x,y):
    
    x = x[:400]
    y = y[:400]

    y[200:400] = 1
    
    train_index, test_index = K_fold(x, 5)
    selected_features = select_feature(x,y, train_index, test_index, 15)
    
    clustering(selected_features,y,10)