import pyedflib
import numpy as np

def load_seizure(f_name, seizure_start, seizure_end):
    
    file_name = f_name
    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)

    label = np.zeros((648,1))
    seizure_start = round(seizure_start/5)
    seizure_end = round(seizure_end/5)
    label[seizure_start:seizure_end] = 1

    data = []
    sigbufs = sigbufs[16:18]
    for chanel in sigbufs:
        data_chanel = []
        for i in range(36,684):
            data_chanel.append(chanel[i*1280:(i+1)*1280])
        data_chanel = np.array(data_chanel)
        data.append(data_chanel)

    data = np.array(data)

    return data, label