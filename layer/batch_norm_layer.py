import numpy as np

class BatchNorm:
    
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 affine: bool = False,
                 train: bool = True) -> None:
        
        self.num_features = num_features
        self.input = None
        self.eps = eps
        
        self.affine = affine
        if (affine):
            self.gamma = np.ones(num_features,)
            self.beta = np.zeros(num_features,)

            self.grad_gamma = np.zeros(self.gamma.shape)
            self.grad_beta = np.zeros(self.beta.shape)
        
        self.train = train
        

    def forward(self, input: np.ndarray, train: bool = True) -> np.ndarray:
        """
        Normalise the whole input using standard normalisation.
        reference: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

        :param input: array of shape (N, d), input to the layer
        :param train: whether in training mode.
        """
        # during testing/evaluation
        # If track_running_stats is set to False, this layer then does not 
        # keep running estimates, and batch statistics are instead used 
        # during evaluation time as well.
        if (not self.train):
            # step 1: subtract mean vector of every training's example
            output = input - self.mean
            
            # step 2: execute normalization
            output = output * self.inv_sqrt_var

            if (self.affine):
                # step 3: multiply with gamma
                output = output @ self.gamma

                # step 4: addition with beta
                output = output + self.beta
            return output
        
        # get the number of examples in the input
        N = input.shape[0]
        
        # normalization step

        # step 1: calculate mean
        self.mean = 1./N * np.sum(input, axis = 0)

        # print(self.mean)

        # step 2: subtract mean vector of every training's example
        self.x_mu = input - self.mean

        # step 3: following the lower branch - calculation denominator
        sq = self.x_mu ** 2

        # step 4: calculate variance
        self.var = 1./N * np.sum(sq, axis = 0)

        # step 5: add eps for numerical stability, then sqrt
        self.sqrt_var = np.sqrt(self.var + self.eps)

        # step 6: invert sqrt_var
        self.inv_sqrt_var = 1. / self.sqrt_var
        
        # step 7: execute normalization
        self.xhat = self.x_mu * self.inv_sqrt_var
        output = self.xhat

        if (self.affine):
            # step 8: multiply with gamma
            self.gamma_x = self.gamma @ self.xhat

            # step 9: addition with beta
            output = self.gamma_x + self.beta

        return output

    def backward(self, delta: np.ndarray = None) -> np.ndarray:
        """
        Propagates delta according to learnable parameters: gamma and beta.
        reference: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

        :param delta: the next layer's derivatives
        """
        N = delta.shape[0]

        # get the derivatives of the learnable parameters
        if (self.affine):

            # step 9
            self.grad_beta = np.sum(delta, axis = 0)
            dgammax = delta

            # step 8
            self.grad_gamma = np.sum(dgammax * self.xhat, axis = 0)
            dxhat = dgammax * self.gamma
            delta = dxhat

        # step 7
        divar = np.sum(delta * self.x_mu, axis = 0)
        dxmu1 = delta * self.inv_sqrt_var

        # step 6
        dsqrtvar = -1. / (self.sqrt_var ** 2) * divar

        # step 5
        dvar = 0.5 * 1. / np.sqrt(self.var + self.eps) * dsqrtvar

        # step 4
        dsq = 1. / N * np.ones(delta.shape) * dvar

        # step 3
        dxmu2 = 2 * self.x_mu * dsq

        # step 2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis = 0)

        # step 1
        dx2 = 1. / N * np.ones(delta.shape) * dmu

        # step 0
        dx = dx1 + dx2

        return dx


if (__name__ == "__main__"):
    pass
