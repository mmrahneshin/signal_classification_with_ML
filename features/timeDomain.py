import numpy as np
from scipy.stats import skew, kurtosis

from features.statistical import rms

def min_max_distance(data):
    return data.max() - data.min()

def skewness(data):
    return skew(data)

def kurtosis_calc(data):
    return kurtosis(data)

def margin_factor(data):
    return data.max() / np.var(data)

def crest_factor(data):
    return data.max() / rms(data)