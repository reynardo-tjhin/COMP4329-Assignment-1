import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from preprocessing import min_max_normalization, standardization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def main():

    # get dataset
    X_train = np.load("./dataset/train_data.npy")
    y_train = np.load("./dataset/train_label.npy")
    X_test = np.load("./dataset/test_data.npy")
    y_test = np.load("./dataset/test_label.npy")

    # load the saved model
    model = MLP([]) # create an empty model
    model = model.load("./best model.zip")

    # perform preprocessing
    # min max normalization was performed
    if (model.preprocessing_type == 'normalization'):
        X_test, _, _ = min_max_normalization(X_test, model.min, model.max)

    # standardization was performed
    elif (model.preprocessing_type == 'standardization'):
        X_test, _, _ = standardization(X_test, model.mean, model.std)

    model.eval()

    # perform inference on training dataset
    y_pred = model.predict(X_train)
    print("train acc={:.5f}%".format(accuracy_score(y_train, y_pred) * 100))
    print("micro precision={:.5f}%".format(precision_score(y_train, y_pred, average='micro') * 100))
    print("macro precision={:.5f}%".format(precision_score(y_train, y_pred, average='macro') * 100))
    print("micro recall={:.5f}%".format(recall_score(y_train, y_pred, average='micro') * 100))
    print("macro recall={:.5f}%".format(recall_score(y_train, y_pred, average='macro') * 100))
    print("micro f1 score={:.5f}%".format(f1_score(y_train, y_pred, average='micro') * 100))
    print("macro f1 score={:.5f}%".format(f1_score(y_train, y_pred, average='macro') * 100))

    print()

    # perform inference on testing dataset
    y_pred = model.predict(X_test)
    print("test acc={:.5f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("micro precision={:.5f}%".format(precision_score(y_test, y_pred, average='micro') * 100))
    print("macro precision={:.5f}%".format(precision_score(y_test, y_pred, average='macro') * 100))
    print("micro recall={:.5f}%".format(recall_score(y_test, y_pred, average='micro') * 100))
    print("macro recall={:.5f}%".format(recall_score(y_test, y_pred, average='macro') * 100))
    print("micro f1 score={:.5f}%".format(f1_score(y_test, y_pred, average='micro') * 100))
    print("macro f1 score={:.5f}%".format(f1_score(y_test, y_pred, average='macro') * 100))

    # confusion matrix on testing dataset
    print()
    cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()
    print(cm)



if (__name__ == "__main__"):
    main()
