from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def confMatrix(x_test, y_test, clf):
    print(clf)
    ConfusionMatrixDisplay.from_estimator(clf, x_test, y_test)
    plt.show()
