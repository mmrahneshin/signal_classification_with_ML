from sklearn.metrics import accuracy_score
from decimal import *
from sklearn import tree


def ID3(x_train, y_train, x_test, y_test):

    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc_score = accuracy_score(y_test,y_pred)

    # print("accuracy score ID3: ", acc_score)

    return acc_score