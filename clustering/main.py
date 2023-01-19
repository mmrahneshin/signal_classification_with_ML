import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

from sklearn.cluster import KMeans

def clustering(x, y, number_of_cluster):
    
    x_train ,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
    random_state=57, shuffle=True)

    # test without clustering ---------------------------------------
    clf = RandomForestClassifier( random_state=57)
    clf.fit(x_train, y_train.ravel())
    pred = clf.predict(x_test)

    acc_score = accuracy_score(y_test, pred)
    rec_score = recall_score(y_test, pred)
    prec_score = precision_score(y_test, pred)

    print("accuracy score : ", acc_score)
    print("recall score : ", rec_score)
    print("precision score : ", prec_score)
    # test without clustering ---------------------------------------

    kmeans = KMeans(n_clusters=number_of_cluster, random_state=57).fit(x_train)

    cluster_classifier = []
    for i in range(number_of_cluster):
        data = x_train[kmeans.labels_ == i]
        label = y_train[kmeans.labels_ == i]

        clf = RandomForestClassifier(random_state=57)
        clf.fit(data, label.ravel())
        cluster_classifier.append(clf)

    y_pred = []
    for test in x_test: 
        cluster = kmeans.predict(test.reshape(1,-1))
        label = cluster_classifier[cluster[0]].predict(test.reshape(1,-1))
        y_pred.append(label)

    y_pred = np.array(y_pred).reshape(-1,1)

    acc_score = accuracy_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    prec_score = precision_score(y_test, y_pred)

    print("accuracy score : ", acc_score)
    print("recall score : ", rec_score)
    print("precision score : ", prec_score)
