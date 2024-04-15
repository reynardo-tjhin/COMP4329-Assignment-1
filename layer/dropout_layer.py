import numpy as np

class Dropout:
    """
    This is an implementation of Dropout.
    Future implementation will be Inverted Dropout.
    """

    # explanation of dropout as a regularization method:
    # https://ai.stackexchange.com/questions/38309/how-does-dropout-work-during-backpropagation

    def __init__(self, probability: float) -> None:

        # probability of dropping the neuron
        self.probability = probability

    def forward(self, input: np.ndarray, train: bool = True) -> np.ndarray:

        # during training
        if (train):
            # step 1: generate an array of random numbers 
            #         of range [0, 1] from Bernoulli distribution
            rand_arr = np.random.uniform(low=0., high=1., size=input.shape)

            # step 2: if the random number generated is greater than the probability,
            #         keep the neuron
            self.mask = rand_arr > self.probability

            # step 3: multiply the mask with the input neuron
            output = self.mask * input

            return output
        
        # during inference
        return self.probability * input


    def backward(self, delta: np.ndarray) -> np.ndarray:
        return self.mask * delta


if (__name__ == "__main__"):
    
    a = np.array([1.5, 4.8, 3.2])
    l = Dropout(probability=0.2).forward(a)