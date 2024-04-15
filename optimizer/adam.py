import numpy as np

from mlp import MLP
from layer.linear_layer import Linear

class Adam:

    def __init__(self, 
                 model: MLP, 
                 lr: float, 
                 rho1: float = 0.9, 
                 rho2: float = 0.999, 
                 epsilon: float = 1e-8) -> None:
        
        self.model = model
        self.lr = lr
        self.rho1 = rho1
        self.rho2 = rho2
        self.epsilon = epsilon
        self.t = 0

        # initialize s
        self.s_w = []
        self.s_b = []
        for layer in self.model.layers:
            if (type(layer) == Linear):
                self.s_w.append(np.zeros_like(layer.W))
                self.s_b.append(np.zeros_like(layer.b))

        # initialize r
        self.r_w = []
        self.r_b = []
        for layer in self.model.layers:
            if (type(layer) == Linear):
                self.r_w.append(np.zeros_like(layer.W))
                self.r_b.append(np.zeros_like(layer.b))

    def step(self):

        index = 0 # to iterate through the linear layers
        self.t += 1
        for layer in self.model.layers:

            if (type(layer) == Linear):

                # step 1: calculate current s (get the s that is specific to the layer)
                self.s_w[index] = self.rho1 * self.s_w[index] + (1 - self.rho1) * layer.grad_W
                self.s_b[index] = self.rho1 * self.s_b[index] + (1 - self.rho1) * layer.grad_b

                # step 2: calculate current r (get the r that is specific to the layer)
                self.r_w[index] = self.rho2 * self.r_w[index] + (1 - self.rho2) * layer.grad_W**2
                self.r_b[index] = self.rho2 * self.r_b[index] + (1 - self.rho2) * layer.grad_b**2

                # step 3: calculate current st
                st_w = self.s_w[index] / (1 - self.rho1**self.t)
                st_b = self.s_b[index] / (1 - self.rho1**self.t)

                # step 4: calculate current rt
                rt_w = self.r_w[index] / (1 - self.rho2**self.t)
                rt_b = self.r_b[index] / (1 - self.rho2**self.t)

                # step 5: update the weights and biases
                layer.W -= self.lr * st_w / np.sqrt(rt_w + self.epsilon)
                layer.b -= self.lr * st_b / np.sqrt(rt_b + self.epsilon)

                # update the index
                index += 1
