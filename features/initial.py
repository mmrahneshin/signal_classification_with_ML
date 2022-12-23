import numpy as np

from features.statistical import *
from features.timeDomain import *

def initialize_features(dataset):
    features_dataset = []
    for data in dataset:
        features = []

        # statistical --------------------------
        features.append(maximum(data))
        features.append(minimum(data))
        features.append(mean(data))
        features.append(variance(data))
        features.append(deviation(data))
        features.append(rms(data))
        # statistical --------------------------

        # timeDomain --------------------------
        features.append(min_max_distance(data))
        features.append(skewness(data))
        features.append(kurtosis_calc(data))
        features.append(margin_factor(data))
        features.append(crest_factor(data))
        # timeDomain --------------------------


        features_dataset.append(features)


    features_dataset = np.array(features_dataset)

    return features_dataset