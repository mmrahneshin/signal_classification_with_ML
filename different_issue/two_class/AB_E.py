import numpy as np
from clustering.main import clustering

def AB_E(x,y):

    x = np.concatenate((x[:200], x[400:500]))
    y = np.concatenate((y[:200], y[400:500]))
    
    clustering(x,y,15)