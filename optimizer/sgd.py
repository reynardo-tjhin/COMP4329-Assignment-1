import numpy as np

from mlp import MLP
from layer.linear_layer import Linear

class SGD:

    def __init__(self, 
                 model: MLP, 
                 lr: float, 
                 momentum: float = 0.9, 
                 weight_decay: float = 0.001) -> None:
        
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # initialize v for momentum
        self.v_w = []
        self.v_b = []
        for layer in self.model.layers:
            if (type(layer) == Linear):
                self.v_w.append(np.zeros_like(layer.W))
                self.v_b.append(np.zeros_like(layer.b))

    def step(self):

        index = 0 # to iterate through the linear layers
        for layer in self.model.layers:

            # update weights and biases
            if (type(layer) == Linear):

                # step 1: calculate v
                self.v_w[index] = self.momentum * self.v_w[index] + self.lr * layer.grad_W
                self.v_b[index] = self.momentum * self.v_b[index] + self.lr * layer.grad_b

                # step 2: calculate weight decay regularizer (only applies to weights)
                w_decay = self.lr * self.weight_decay * layer.W

                # step 3: update
                layer.W = layer.W - self.v_w[index] - w_decay
                layer.b = layer.b - self.v_b[index]

                # update index
                index += 1

                # old method
                # layer.W -= self.lr * layer.grad_W
                # layer.b -= self.lr * layer.grad_b