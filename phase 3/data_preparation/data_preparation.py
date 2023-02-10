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

    seizure_index = [index for index in range(len(temp_label)) if temp_label[index] == 1]

    normal_index = [index for index in range(len(temp_label)) if temp_label[index] == 0]
    normal_index = np.random.choice(normal_index, size=len(seizure_index))

    label = temp_label[np.concatenate((seizure_index,normal_index))]
    data = temp_data[:,np.concatenate((seizure_index,normal_index))]


    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb01_04.edf",
    1467, 1494)
    data, label = seizure_pre(temp_data, temp_label, data, label)



    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb01_15.edf", 
    1732, 1772)
    data, label = seizure_pre(temp_data, temp_label, data, label)



    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb01_16.edf", 
    1015, 1066)
    data, label = seizure_pre(temp_data, temp_label, data, label)

    
    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb01_18.edf", 
    1720, 1810)
    data, label = seizure_pre(temp_data, temp_label, data, label)



    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb01_21.edf", 
    327, 420)
    data, label = seizure_pre(temp_data, temp_label, data, label)



    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb01_26.edf", 
    1862, 1963)
    data, label = seizure_pre(temp_data, temp_label, data, label)



    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb02_16.edf", 
    130, 212)
    data, label = seizure_pre(temp_data, temp_label, data, label)



    temp_data, temp_label = load_seizure("./phase 3/data_preparation/load_data/seizure/data/chb02_16+.edf", 
    2972, 3053)
    data, label = seizure_pre(temp_data, temp_label, data, label)
    #load seizure ---------------------------------------------------------

    print(data.shape, label.shape)

    #load normal ---------------------------------------------------------
    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_01.edf")
    data, label = normal_pre(temp_data, temp_label, data, label, 200)



    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_02.edf")
    data, label = normal_pre(temp_data, temp_label, data, label, 200)



    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_05.edf")
    data, label = normal_pre(temp_data, temp_label, data, label, 200)



    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_06.edf")
    data, label = normal_pre(temp_data, temp_label, data, label, 200)


    
    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_07.edf")
    data, label = normal_pre(temp_data, temp_label, data, label, 200)



    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_08.edf")
    data, label = normal_pre(temp_data, temp_label, data, label, 200)



    temp_data, temp_label = load_normal("./phase 3/data_preparation/load_data/normal/data/chb01_09.edf")
    data, label = normal_pre(temp_data, temp_label, data, label, 200)
    #load normal ---------------------------------------------------------

    print(np.sum(label))
    return data, label


def seizure_pre(temp_data, temp_label, data, label):
    seizure_index = [index for index in range(len(temp_label)) if temp_label[index] == 1]

    normal_index = [index for index in range(len(temp_label)) if temp_label[index] == 0]
    normal_index = np.random.choice(normal_index, size=len(seizure_index))


    temp_label = temp_label[np.concatenate((seizure_index,normal_index))]
    temp_data = temp_data[:,np.concatenate((seizure_index,normal_index))]

    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    return data, label


def normal_pre(temp_data, temp_label, data, label, length):
    normal_index = np.arange(len(temp_label))
    normal_index = np.random.choice(normal_index, size=length)
    
    temp_label = temp_label[normal_index]
    temp_data = temp_data[:,normal_index]

    data = np.append(data, temp_data, axis=1)
    label = np.append(label, temp_label, axis=0)

    return data, label