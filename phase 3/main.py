import numpy as np

from data_preparation.data_preparation import data_preparation
from cnn.cnn import cnn

import random
import os
seed = 57

def main():
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


    data, label = data_preparation()
    print(data.shape, label.shape)
    
    cnn(data,label)

main()