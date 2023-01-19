from clustering.main import clustering

def AB_CD(x,y):
    
    x = x[:400]
    y = y[:400]

    y[200:400] = 1
    
    clustering(x,y,3)