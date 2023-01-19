from clustering.main import clustering

def AB_CD_E(x,y):
    
    y[:200] = 1

    clustering(x,y, 10)