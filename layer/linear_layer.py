import numpy as np

class Linear:

    def __init__(self, n_in: int, n_out: int, W=None, b=None) -> None:
        
        self.n_in = n_in
        self.n_out = n_out

        if (type(W) is np.ndarray):
            self.W = W
        else: # assuming the weight given is None
            self.W = np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out),
            )
        
        if (type(b) is np.ndarray):
            self.b = b
        else: # assuming the biases given is None
            self.b = np.zeros(self.n_out,)

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, input, train: bool = True) -> np.ndarray:
        self.input = input
        output = input @ self.W + self.b
        return output

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        :return delta of shape (N, n_in)
        """
        batch_size = np.atleast_2d(delta).shape[0] # number of examples in a single batch

        # gradient of weights = delta * input
        # Weights (n_in,n_out) = (n_in,) * (,n_out)
        self.grad_W = np.atleast_2d(self.input).T @ np.atleast_2d(delta) / batch_size
        # self.grad_W = np.vstack([delta] * self.n_in) * np.vstack([self.input] * self.n_out).T
        
        # gradient of beta = delta
        if (len(delta.shape) == 1): # a single example training
            self.grad_b = delta
        else:
            self.grad_b = np.sum(delta, axis=0) / batch_size

        # delta's shape: (N, n_in)
        self.delta = delta @ self.W.T

        return self.delta


# tests
def test1():
    """Stochastic (single example)"""

    # define input, weight, bias and linear layer
    x = np.array([1.5, 4.8, 3.2, 5.6, 9.4])
    w = np.array([
        [-0.49,  0.75, -0.09],
        [ 0.62, -0.25, -0.80],
        [-0.86, -0.31,  0.11],
        [ 0.67,  0.64,  0.24],
        [-0.19,  0.55, -0.07],
    ])
    b = np.array([0.11, 0.09, 0.06])
    l = Linear(n_in=5, n_out=3, W=w, b=b)

    # testing for forward
    # print("testing for forward:")
    output = x @ w + b
    l_output = l.forward(x)
    assert np.allclose(output, l_output)

    # delta from the previous layer of backpropagation
    delta = np.array([0.25, 0.17, 0.12])

    # testing for backward
    # print("testing for backward:")
    l_delta = l.backward(delta)

    # testing for gradient of biases
    grad_b = delta
    assert np.allclose(grad_b, l.grad_b)

    # testing for gradient of weights
    grad_W = np.zeros((5, 3))
    grad_W[0][0] = delta[0]*x[0]; grad_W[0][1] = delta[1]*x[0]; grad_W[0][2] = delta[2]*x[0]
    grad_W[1][0] = delta[0]*x[1]; grad_W[1][1] = delta[1]*x[1]; grad_W[1][2] = delta[2]*x[1]
    grad_W[2][0] = delta[0]*x[2]; grad_W[2][1] = delta[1]*x[2]; grad_W[2][2] = delta[2]*x[2]
    grad_W[3][0] = delta[0]*x[3]; grad_W[3][1] = delta[1]*x[3]; grad_W[3][2] = delta[2]*x[3]
    grad_W[4][0] = delta[0]*x[4]; grad_W[4][1] = delta[1]*x[4]; grad_W[4][2] = delta[2]*x[4]
    assert np.allclose(grad_W, l.grad_W)

    # testing for delta
    n_delta = np.zeros(5)
    n_delta[0] = delta @ w[0].T
    n_delta[1] = delta @ w[1].T
    n_delta[2] = delta @ w[2].T
    n_delta[3] = delta @ w[3].T
    n_delta[4] = delta @ w[4].T
    assert np.allclose(n_delta, l_delta)

    print("Test 1: Success!")


def test2():
    """Testing for mini-batch training"""
    
    # define input, weight, bias and linear layer
    # 8 instances, 5 attributes
    x = np.array([
        [1.5, 4.8, 3.2, 5.6, 9.4],
        [9.1, 8.0, 7.2, 0.1, 3.1],
        [3.6, 0.9, 3.7, 6.8, 9.2],
        [5.3, 7.5, 4.7, 2.7, 2.8],
        [6.2, 4.8, 3.4, 6.5, 8.1],
        [2.9, 6.7, 3.1, 8.0, 0.7],
        [1.2, 2.5, 9.7, 8.1, 4.8],
        [4.2, 1.5, 4.5, 1.6, 2.0],
    ])
    # n_in=5, n_out=3
    w = np.array([ 
        [-0.49,  0.75, -0.09],
        [ 0.62, -0.25, -0.80],
        [-0.86, -0.31,  0.11],
        [ 0.67,  0.64,  0.24],
        [-0.19,  0.55, -0.07],
    ])
    b = np.array([0.11, 0.09, 0.06])
    l = Linear(5, 3, W=w, b=b)

    # testing for forward
    o = x @ w + b
    assert np.allclose(o, l.forward(x))

    # testing for backward
    # previous delta's shape: (8, 3) aka (batch_size, n_out)
    delta = np.array([
        [6.03, 7.51, 7.46],
        [6.51, 7.01, 5.94],
        [5.01, 3.20, 4.38],
        [2.61, 6.69, 9.49],
        [4.53, 7.82, 5.21],
        [7.29, 2.06, 8.85],
        [1.71, 6.29, 1.78],
        [9.03, 6.24, 6.91],
    ])
    l_delta = l.backward(delta)
    n_delta = delta @ w.T
    assert np.allclose(n_delta, l_delta)

    # testing for grad b
    assert np.allclose(np.sum(delta, axis=0) / 8, l.grad_b)

    # testing for grad W
    grad_w = x.T @ delta / 8 # where 8 is the batch size
    assert np.allclose(grad_w, l.grad_W)

    print("Test 2: Success!")


if (__name__ == "__main__"):

    # set print options to horizontal
    oldoptions = np.get_printoptions()
    np.set_printoptions(linewidth=np.inf)

    test1()
    test2()

    # revert numpy print options
    np.set_printoptions(**oldoptions)
