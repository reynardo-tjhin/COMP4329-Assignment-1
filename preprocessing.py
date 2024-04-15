import numpy as np

from typing import Tuple

def one_hot(y: np.ndarray, n_classes: int = None) -> np.ndarray:
    """
    Perform a one-hot encoding on the label data.
    """
    # ensure that the label data is of index, i.e. integers
    y = np.array(y).astype(np.int64)

    # if number of classes is given
    if (n_classes):
        one_hot = np.zeros((y.shape[0], n_classes))
    else:
        one_hot = np.zeros((y.shape[0], y.max() + 1))

    # change the data to 1 based on the label
    one_hot[np.arange(y.shape[0]), y] = 1

    return one_hot


def min_max_normalization(x: np.ndarray,
                          minimum: np.ndarray = None,
                          maximum: np.ndarray = None) -> np.ndarray:
    """
    Min-Max Normalization (0-1 range).
    """
    # step 1: if minimum is not given, calculate the minimum of each attribute
    if (type(minimum) != np.ndarray):
        minimum = np.min(x, axis=0)

    # step 2: if maximum is not given, calculate the maximum of each attribute
    if (type(maximum) != np.ndarray):
        maximum = np.max(x, axis=0)

    # step 3: perform min max normalization
    normalized_x = (x - minimum) / (maximum - minimum)

    return normalized_x, minimum, maximum


def standardization(x: np.ndarray, 
                    mean: np.ndarray = None,
                    standard_deviation: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardization preprocessing.
    """
    # step 1: if mean is not given, calculate the mean of each attribute/feature
    if (type(mean) != np.ndarray):
        mean = np.mean(x, axis=0)

    # step 2: if std is not given, calculate the standard deviation of each attribute/feature
    if (type(standard_deviation) != np.ndarray):
        standard_deviation = np.std(x, axis=0)

    # step 3: perform standardization
    standardized_x = (x - mean) / standard_deviation
    
    return standardized_x, mean, standard_deviation

# for any testing
if (__name__ == "__main__"):
    pass