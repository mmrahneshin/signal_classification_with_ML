from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

def roc_matrix(x_test, y_test, clf):
    print(clf)
    RocCurveDisplay.from_estimator(clf, x_test, y_test)
    plt.show()
