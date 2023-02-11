import numpy as np

from cnn.select_features.featureAccuracy import feature_accuracy
from cnn.select_features.correlation import correlation

features_name = ["max", "min", "mean" , "variance", "deviation"
    , "rms", "min_max_distance", "skewness" , "kurtosis" , "margin_factor"
    ,"crest_factor", "max_f", "min_f", "mean_f", "var_f", "margin_factor_f", "peak_f"]

def select_feature(x, y, train_index, test_index, count):
    x_features = x.T

    feature_score = feature_accuracy(x, y, train_index, test_index)
    feature_score = feature_score.reshape(-1, 1)

    mean_corr = correlation(x)
    mean_corr = mean_corr.reshape(-1, 1)
    
    # normalize -------------------------------------------
    
    # normalize -------------------------------------------
    f1_score = 2 * feature_score * (1 / mean_corr) / ((1 / mean_corr) + feature_score)

    feature_dict = dict()
    for name,f1 in zip(features_name, f1_score):
        feature_dict[name] = f1[0]

    sorted_features_by_f1 = sorted(feature_dict.items(), key=lambda x: x[1] , reverse=True)

    selected_features = []

    for i in range(count):
        selected_features.append(x_features[features_name.index(sorted_features_by_f1[i][0])])

    selected_features = np.array(selected_features)

    return selected_features.T
