from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.svm import SVC
from decimal import *


def SVM(x_train, y_train, x_test, y_test):
    
    clf = SVC()
    clf.fit(x_train, y_train)    
    y_pred = clf.predict(x_test)

    acc_score = accuracy_score(y_test,y_pred)
    rec_score = recall_score(y_test,y_pred)
    prec_score = precision_score(y_test,y_pred)

    print("accuracy score SVM: ", acc_score)
    print("recall score SVM: ", rec_score)
    print("precision score SVM: ", prec_score)

    return acc_score, clf
