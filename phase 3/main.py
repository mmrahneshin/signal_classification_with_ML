import numpy as np

from data_preparation.data_preparation import data_preparation
from cnn.cnn import cnn1,cnn2,cnn3

import random
import os
seed = 57

def main():
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


    data, label = data_preparation()
    print(data.shape, label.shape)
    
    cnn1(data,label)

    # cnn2(data, label)

    # cnn3(data, label)


main()