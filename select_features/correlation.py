import numpy as np

def correlation(x):
    x_features = x.T
   
    correlation_matrix = []
    for feature1 in x_features:
        row = []
        for feature2 in x_features:
            row.append(np.corrcoef(feature1,feature2)[0,1])

        row = np.array(row)

        correlation_matrix.append(row)

    correlation_matrix = np.array(correlation_matrix)


    return correlation_matrix