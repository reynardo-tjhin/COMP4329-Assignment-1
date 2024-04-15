import numpy as np
import time
import argparse

from mlp import MLP
from layer.linear_layer import Linear
from layer.activation_layer import Activation
from layer.dropout_layer import Dropout
from layer.batch_norm_layer import BatchNorm
from criterion.mse import MSELoss
from criterion.softmax_and_cce import SoftmaxAndCCELoss
from optimizer.sgd import SGD
from optimizer.adam import Adam
from preprocessing import standardization, min_max_normalization, one_hot

from sklearn.metrics import accuracy_score

def main(args: argparse.Namespace):

    # get dataset
    X_train = np.load("./dataset/train_data.npy")
    y_train = np.load("./dataset/train_label.npy")
    X_test = np.load("./dataset/test_data.npy")
    y_test = np.load("./dataset/test_label.npy")

    # perform one hot encoding
    one_hot_y_train = one_hot(y_train, n_classes=10)

    preprocessing = args.preprocessing
    if (preprocessing == 'standardization'):
        X_train, X_train_mean, X_train_std = standardization(X_train)
        X_test, _, _ = standardization(
            X_test, 
            mean=X_train_mean, 
            standard_deviation=X_train_std,
        )
    elif (preprocessing == 'normalization'):
        X_train, X_train_min, X_train_max = min_max_normalization(X_train)
        X_test, _, _ = min_max_normalization(
            X_test, 
            minimum=X_train_min, 
            maximum=X_train_max,
        )

    print(X_train.shape)
    print(one_hot_y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # create model for dataset
    if (args.model == 'default'):
        # First model: 128 -> 80 -> relu -> 40 -> relu -> 10 -> tanh
        model = MLP(
            layers=[
                Linear(128, 80),
                Activation('relu'),
                Linear(80, 40),
                Activation('relu'),
                Linear(40, 10),
                Activation('tanh'),
            ]
        )
    elif (args.model == 'single'):
        # with a single hidden layer
        model = MLP(
            layers=[
                Linear(128, 90),
                Activation('relu'),
                Linear(90, 10),
                Activation('tanh'),
            ]
        )
    elif (args.model == 'dropout'):
        # with dropout layer
        model = MLP(
            layers=[
                Linear(128, 80),
                Activation('relu'),
                Linear(80, 40),
                Activation('relu'),
                Dropout(0.5),
                Linear(40, 10),
                Activation('tanh'),
            ]
        )
    elif (args.model == 'batchnorm'):
        # with batch norm layer
        # with dropout layer
        model = MLP(
            layers=[
                Linear(128, 80),
                BatchNorm(80),
                Activation('relu'),
                Linear(80, 40),
                BatchNorm(40),
                Activation('relu'),
                Linear(40, 10),
                BatchNorm(10),
                Activation('tanh'),
            ]
        )
    if (preprocessing == 'standardization'):
        model.save_preprocessing_parameters(
            type='standardization',
            mean=X_train_mean,
            std=X_train_std,
        )
    elif (preprocessing == 'normalization'):
        model.save_preprocessing_parameters(
            type='normalization', 
            min=X_train_min,
            max=X_train_max,
        )

    # for mini-batch training: currently only apply for GD
    # batch_size = 1: stochastic gradient descent
    # batch_size = n (where 1 < n < total_examples): mini batch training
    # batch_size = total_examples: batch gradient descent
    # Adam prefers stochastic
    n_examples = X_train.shape[0]
    batch_size = args.batch_size
    no_of_batches = int(np.ceil(n_examples/batch_size))

    # state the number of epochs
    epochs = args.epochs

    # define loss function
    if (args.loss_function == 'mse'):
        loss_fn = MSELoss()
    elif (args.loss_function == 'softmax_and_cce'):
        loss_fn = SoftmaxAndCCELoss()
    
    # define optimizer
    if (args.optimizer == 'sgd'):
        optimizer = SGD(
            model=model,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif (args.optimizer == 'adam'):
        optimizer = Adam(
            model=model,
            lr=args.learning_rate,
        )

    # train the model
    start = time.time()
    for epoch in range(epochs):

        # indexes for batch training
        start_index = 0
        end_index = (start_index + batch_size) if (start_index + batch_size < n_examples) else n_examples

        # batch training
        model.train()
        total_loss = 0
        for _ in range(no_of_batches):

            # get the batches
            batch_x = X_train[start_index:end_index]
            batch_y = one_hot_y_train[start_index:end_index]
            
            # forward pass
            y_hat = model.forward(batch_x)

            # backward pass
            batch_loss, delta = loss_fn.loss_fn(batch_y, y_hat)
            model.backward(delta)

            # update
            optimizer.step()

            # update the batch index
            start_index = end_index
            if (start_index + batch_size < n_examples):
                end_index = start_index + batch_size
            else:
                end_index = n_examples

            # add the loss
            total_loss += np.sum(batch_loss) # batch loss is a single dimensional array

        mean_loss = total_loss / n_examples
        print("[%3d/%3d]: loss=%.16f" % (epoch + 1, epochs, mean_loss), end=",")

        # inference/testing/evaluation
        model.eval()
        y_pred = model.predict(X_train)
        print("train_acc={:.5f}%".format(accuracy_score(y_train, y_pred) * 100), end=",")

        y_pred = model.predict(X_test)
        print("test_acc={:.5f}%".format(accuracy_score(y_test, y_pred) * 100))

    end = time.time()
    print("time taken: %.5f seconds" % (end - start))

    # save the model
    model.save("model")


if (__name__ == "__main__"):

    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', 
                        default=32, 
                        type=int)
    parser.add_argument('-e', '--epochs',
                        default=50,
                        type=int)
    parser.add_argument('-o', '--optimizer', 
                        choices=['sgd', 'adam'], 
                        default='adam', 
                        type=str)
    parser.add_argument('-lr', '--learning_rate', 
                        default=0.01, 
                        type=float)
    parser.add_argument('-m', '--momentum', 
                        default=0., 
                        type=float)
    parser.add_argument('-w_decay', '--weight_decay', 
                        default=0., 
                        type=float)
    parser.add_argument('-l', '--loss_function',
                        choices=['mse', 'softmax_and_cce'],
                        default='softmax_and_cce',
                        type=str)
    parser.add_argument('-p', '--preprocessing',
                        choices=['normalization', 'standardization'],
                        default='standardization',
                        type=str)
    
    # TEMP (for experimenting)
    parser.add_argument('-mdl', '--model',
                        choices=['default', 'single', 'dropout', 'batchnorm'],
                        default='default',
                        type=str)
    
    args = parser.parse_args()

    main(args)
