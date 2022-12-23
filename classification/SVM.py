from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.svm import SVC
from decimal import *


def SVM(x_train, y_train, x_test, y_test):
    
    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)    
    y_pred = clf.predict(x_test)

    print("accuracy score: ", Decimal(accuracy_score(y_test,y_pred)))
    print("recall score: ", Decimal(recall_score(y_test,y_pred)))
    print("precision score: ", Decimal(precision_score(y_test,y_pred)))
