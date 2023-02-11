import numpy as np

def maximum(data):
    return data.max()

def minimum(data):
    return data.min()

#average
def mean(data):
    return np.mean(data)

def variance(data):
    return np.var(data)

def deviation(data):
    return np.std(data)

# root mean square
def rms(data):
    return np.sqrt(np.mean(np.square(data)))