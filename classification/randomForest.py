from sklearn.metrics import accuracy_score, recall_score, precision_score
from decimal import *
from sklearn.ensemble import RandomForestClassifier

def random_forest(x_train, y_train,x_test,y_test):
    
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc_score = accuracy_score(y_test,y_pred)
    rec_score = recall_score(y_test,y_pred)
    prec_score = precision_score(y_test,y_pred)

    print("accuracy score: ", Decimal(acc_score))
    print("recall score: ", Decimal(rec_score))
    print("precision score: ", Decimal(prec_score))

    return acc_score, clf
