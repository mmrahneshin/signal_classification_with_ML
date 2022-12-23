import numpy as np

from features.statistical import *

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

        features_dataset.append(features)


    features_dataset = np.array(features_dataset)

    return features_dataset