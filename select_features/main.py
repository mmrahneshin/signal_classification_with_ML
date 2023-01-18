import numpy as np

from select_features.featureAccuracy import feature_accuracy
from select_features.correlation import correlation


features_name = ["max", "min", "mean" , "variance", "deviation"
    , "rms", "min_max_distance", "skewness" , "kurtosis" , "margin_factor"
    ,"crest_factor", "max_f", "min_f", "mean_f", "var_f", "margin_factor_f", "peak_f"]

def select_feature(x, y, train_index, test_index):
    feature_score = feature_accuracy(x, y, train_index, test_index)
    correlation_matrix = correlation(x)
