from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def confMatrix(y_test, y_pred, label):
    cm = confusion_matrix(y_test, y_pred, labels=label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=label)
    disp.plot()
