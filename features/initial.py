import numpy as np
from scipy.fft import fft

from features.statistical import *
from features.timeDomain import *
from features.frequencyDomain import *

def initialize_features(dataset):
    features_dataset = []
    for data in dataset:
        features = []

        # statistical --------------------------
        features.append(maximum(data))              #1
        features.append(minimum(data))              #2
        features.append(mean(data))                 #3
        features.append(variance(data))             #4
        features.append(deviation(data))            #5
        features.append(rms(data))                  #6
        # statistical --------------------------

        # timeDomain --------------------------
        features.append(min_max_distance(data))     #7
        features.append(skewness(data))             #8
        features.append(kurtosis_calc(data))        #9
        features.append(margin_factor(data))        #10
        features.append(crest_factor(data))         #11
        # timeDomain --------------------------

        # frequencyDomain --------------------------
        ft = fft(data)
        S = np.abs(ft**2)/len(data)

        features.append(max_f(S))                #12
        features.append(min_f(S))                #13
        features.append(mean_f(S))               #14
        features.append(var_f(S))                #15
        features.append(margin_factor_f(S))      #16
        features.append(peak_f(S))               #17
        # frequencyDomain --------------------------



        features_dataset.append(features)


    features_dataset = np.array(features_dataset)

    return features_dataset