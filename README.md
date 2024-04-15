# COMP4329-Assignment-1

## Marked Modules

Modules to be marked can be found in:
- More than one hidden layer: `main.py` which is indicated by more than 2 'Linear' layers.
- ReLU activation: `./layer/activation_layer.py`; in Activation class.
- Weight decay: `./optimizer/sgd.py`; in the second step of `step()` function.
- Momentum in SGD: `./optimizer/sgd.py`; in the first and third step of `step()` function.
- Dropout: `./layer/dropout_layer.py`
- Softmax and Cross-Entropy Loss: `./criterion/softmax_and_cce.py`
- Mini-batch training: `main.py`; a mini-batch size can be given.
- Batch Normalization: `./layer/batch_norm_layer.py`
- Other advanced operation (Adam): `./optimizer/adam.py`

## How to run each module

To perform individual (or unit) tests on each model, simply go to the folder of interest and run `python3 [file]`.
Some of the modules have individual tests on them, for example in the `activation_layer.py`, the tests are performed
against each activation functions and its derivatives.

## How to train

**Note: please change the dataset folder to `dataset/` because the `main.py` file only recognizes the dataset with the folder name of `dataset/`**

1. Go to `run.sh`
2. Change the hyperparameters as you want
3. Run the file by typing `./run.sh` in the terminal

If the `bash: ./run.sh: Permission denied` error occurs, run `chmod +x run.sh` to give permission to run the file.

## How to run trained model

Simply run `python3 load_saved_model.py`, it will load the `best_model.zip` and perform performance metrics on the model.

## Other files/folder

- `./clear.sh`: clear temporary python files
- `./run_experiment.sh`: run the different experiments for analysis
- `./notebooks/`: to create plots for analysis/evaluation and for learning each module

## Libraries Required

- numpy
- sk-learn (for getting the metrics)
