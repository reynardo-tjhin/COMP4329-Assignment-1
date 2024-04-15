import numpy as np

from mlp import MLP
from preprocessing import min_max_normalization, standardization
from sklearn.metrics import accuracy_score

def main():

    # get dataset
    X_test = np.load("./dataset/test_data.npy")
    y_test = np.load("./dataset/test_label.npy")

    # load the saved model
    model = MLP([]) # create an empty model
    model = model.load("./saved_models/mini-batch-1/1_1_model.zip")

    # perform preprocessing
    # min max normalization was performed
    if (model.preprocessing_type == 'normalization'):
        X_test, _, _ = min_max_normalization(X_test, model.min, model.max)

    # standardization was performed
    elif (model.preprocessing_type == 'standardization'):
        X_test, _, _ = standardization(X_test, model.mean, model.std)

    # perform inference on testing dataset
    model.eval()
    y_pred = model.predict(X_test)
    print("test_acc={:.5f}%".format(accuracy_score(y_test, y_pred) * 100))


if (__name__ == "__main__"):
    main()
