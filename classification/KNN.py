from sklearn.metrics import accuracy_score, recall_score, precision_score
from decimal import *
from sklearn.neighbors import KNeighborsClassifier


def KNN(x_train, y_train, x_test, y_test):

    neigh = KNeighborsClassifier()
    neigh.fit(x_train, y_train)
    y_pred = neigh.predict(x_test)

    print("accuracy score: ", Decimal(accuracy_score(y_test,y_pred)))
    print("recall score: ", Decimal(recall_score(y_test,y_pred)))
    print("precision score: ", Decimal(precision_score(y_test,y_pred)))