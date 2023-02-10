import pyedflib
import numpy as np

def load_normal(f_name):
    
    file_name = f_name
    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)

    label = np.zeros((int((f.getNSamples()[0]/256) - (f.getNSamples()[0]/256)/10 - 5), 1))

    data = []
    sigbufs = sigbufs[16:18]
    for chanel in sigbufs:
        data_chanel = []
        for i in range(int((f.getNSamples()[0]/256)/20) , int((f.getNSamples()[0]/256) - (f.getNSamples()[0]/256)/20 - 5)):
            data_chanel.append(chanel[i*256:(i+5)*256])
        data_chanel = np.array(data_chanel)
        data.append(data_chanel)

    data = np.array(data)

    return data, label