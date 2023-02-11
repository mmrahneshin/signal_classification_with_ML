import numpy as np

def max_f(data):
    return data.max()

def min_f(data):
    return data.min()

def mean_f(data):
    return np.mean(data)

def var_f(data):
    return np.var(data)

def margin_factor_f(data):
    return data.max() / np.var(data) 

def peak_f(data):
    return np.max(np.abs(data))
