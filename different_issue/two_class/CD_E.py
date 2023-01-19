from clustering.main import clustering

def CD_E(x,y):

    x = x[200:500]
    y = y[200:500]
    
    clustering(x,y,10)