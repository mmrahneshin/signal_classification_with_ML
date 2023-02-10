import numpy as np
from data_preparation.load_data.seizure.load_seizure import load_seizure
from data_preparation.load_data.normal.load_normal import load_normal

def data_preparation():
    data = []
    label = []

    temp_data = []
    temp_label = []
    #load seizure ---------------------------------------------------------
    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb01_03.edf", 
    2996, 3036)
    data = temp_data
    label = temp_label

    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb01_04.edf",
    1467, 1494)
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb01_15.edf", 
    1732, 1772)
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb01_16.edf", 
    1015, 1066)
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)
    
    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb01_18.edf", 
    1720, 1810)
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb01_21.edf", 
    327, 420)
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)
    #load seizure ---------------------------------------------------------


    #load normal ---------------------------------------------------------
    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_01.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_02.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_05.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_06.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)
    
    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_07.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_08.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_09.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_10.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_11.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_12.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_13.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_14.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_17.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_19.edf")
    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    #load normal ---------------------------------------------------------

    return data, label