import numpy as np

class SoftmaxAndCCELoss:

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray):

        # ASSUMPTION: y is already one-hot encoded!
        
        # forward
        # step 1: get the softmax output
        softmax_output = self._softmax(y_hat)

        # step 2: calculate the cross-entropy loss
        loss = self._cce_loss(softmax_output, y)

        # backward        
        # step 1: calculate delta wrt softmax_output
        cce_delta = self._cce_delta(y, softmax_output)

        # step 2: get the derivative of softmax
        softmax_deriv = self._softmax_derivative(softmax_output)

        # step 3: get the delta
        delta = self._softmax_cce_loss_delta(cce_delta, softmax_deriv)

        return loss, delta


    def _softmax(self, y_hat: np.ndarray) -> np.ndarray:

        # for numerical stability: subtract the y_hat by the maximum value across the row
        # single dimension
        if (len(y_hat.shape) == 1):
            exp_y_hat = np.exp(y_hat - np.max(y_hat))
            return exp_y_hat / np.sum(exp_y_hat, axis=0)

        # two dimensions
        exp_y_hat = np.exp(y_hat - np.max(y_hat, axis=1).reshape((-1,1)))
        return exp_y_hat / np.sum(exp_y_hat, axis=1).reshape((-1,1))
    

    def _softmax_derivative(self, softmax: np.ndarray) -> np.ndarray:
        """
        :returns a Jacobian matrix.
        """
        # reference: https://stackoverflow.com/questions/45949141/compute-a-jacobian-matrix-from-scratch-in-python
        # single dimension
        if (len(softmax.shape) == 1):
            deriv = softmax.reshape(-1, 1)
            return np.diagflat(deriv) - np.dot(deriv, deriv.T)

        # two dimensions
        batch_size = softmax.shape[0]
        output_size = softmax.shape[1]
        deriv = np.zeros((batch_size, output_size, output_size))
        for i in range(batch_size):
            temp = softmax[i].reshape(-1, 1)
            deriv[i] = np.diagflat(temp) - np.dot(temp, temp.T)
        return deriv
    

    def _cce_loss(self, input: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> np.float64:
        
        # for numerical stability: add a very small epsilon 
        # value to the predicted probability output (avoid np.log(0.))
        # single dimension
        if (len(target.shape) == 1):
            return -np.sum(target * np.log(input + eps))

        # two dimensions
        return -np.sum(target * np.log(input + eps), axis=1)
    

    def _cce_delta(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return -y / p
    

    def _softmax_cce_loss_delta(self, cce_delta: np.ndarray, softmax_deriv: np.ndarray) -> np.ndarray:
        
        # single dimension
        if (len(cce_delta.shape) == 1):
            return cce_delta @ softmax_deriv
    
        # two dimensions cce_delta
        batch_size = cce_delta.shape[0]
        output_size = cce_delta.shape[1]
        delta = np.zeros((batch_size, output_size))
        for i in range(batch_size):
            delta[i] = cce_delta[i] @ softmax_deriv[i]
        return delta


# testing
def test1():
    """testing for a single example (or single array)"""
        
    # categorical
    a = np.array([1.5, 4.8, 3.2]) # y_hat
    o = np.array([0., 1., 0.]) # y

    # testing softmax
    softmax_output = np.zeros((3,))
    softmax_output[0] = np.exp(1.5) / (np.exp(1.5) + np.exp(4.8) + np.exp(3.2))
    softmax_output[1] = np.exp(4.8) / (np.exp(1.5) + np.exp(4.8) + np.exp(3.2))
    softmax_output[2] = np.exp(3.2) / (np.exp(1.5) + np.exp(4.8) + np.exp(3.2))
    assert np.allclose(softmax_output, SoftmaxAndCCELoss()._softmax(a))

    # testing cce loss
    eps = 1e-9
    cce_loss = -o[0]*np.log(softmax_output[0] + eps) \
                    - o[1]*np.log(softmax_output[1] + eps) \
                    - o[2]*np.log(softmax_output[2] + eps)
    assert cce_loss == SoftmaxAndCCELoss()._cce_loss(softmax_output, o)

    # testing cce delta
    cce_delta = np.zeros((3,))
    cce_delta[0] = -o[0] / softmax_output[0]
    cce_delta[1] = -o[1] / softmax_output[1]
    cce_delta[2] = -o[2] / softmax_output[2]
    assert np.allclose(cce_delta, SoftmaxAndCCELoss()._cce_delta(o, softmax_output))
    
    # testing softmax derivative
    deriv = softmax_output * (1. - softmax_output)
    deriv = np.diagflat(deriv)
    deriv[0][1] = -softmax_output[1] * softmax_output[0]
    deriv[0][2] = -softmax_output[2] * softmax_output[0]
    deriv[1][0] = -softmax_output[0] * softmax_output[1]
    deriv[1][2] = -softmax_output[2] * softmax_output[1]
    deriv[2][0] = -softmax_output[0] * softmax_output[2]
    deriv[2][1] = -softmax_output[1] * softmax_output[2]
    assert np.allclose(deriv, SoftmaxAndCCELoss()._softmax_derivative(softmax_output))

    # testing delta
    delta = cce_delta @ deriv
    _, l_delta = SoftmaxAndCCELoss().loss_fn(o, a)
    assert np.allclose(delta, l_delta)

    print("Test 1: Success!")

def test2():
    """testing on a single batch, for mini-batch training"""

    # 8 instances, 5 classes
    a = np.array([
        [0.15994562, 7.57540417, 2.74826289, 6.31954184, 9.38727067],
        [5.34793979, 1.79035884, 2.13777807, 5.45324409, 0.95563084],
        [6.58595483, 3.53615929, 3.59721189, 8.90591486, 0.49245171],
        [5.57384823, 0.86206866, 4.25205943, 9.46046715, 5.6539647 ],
        [1.46855443, 2.94314283, 1.7716562 , 9.34939117, 2.00402701],
        [3.72361313, 7.56615657, 1.08628823, 9.83534009, 8.97623911],
        [7.62197105, 8.27190668, 2.21428423, 8.62613788, 3.78708596],
        [4.26175126, 7.4891155 , 0.27106331, 0.77707443, 1.39126848],
    ])
    o = np.array([
        [0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0.],
    ])

    # testing softmax
    softmax_output = np.zeros((8, 5))
    for i in range(softmax_output.shape[0]):
        exp_y_hat = np.exp(a[i] - np.max(a[i]))
        softmax_output[i] = exp_y_hat / np.sum(exp_y_hat, axis=0)
        assert np.allclose(np.sum(softmax_output[i]), np.array([1.]))
    assert np.allclose(softmax_output, SoftmaxAndCCELoss()._softmax(a))

    # testing cce loss
    eps = 1e-9
    cce_loss = np.zeros((8,)) # 8 losses
    for i in range(a.shape[0]):
        cce_loss[i] = -np.sum(o[i] * np.log(softmax_output[i] + eps), axis=0)    
    assert np.allclose(cce_loss, SoftmaxAndCCELoss()._cce_loss(softmax_output, o))

    # testing cce delta
    cce_delta = np.zeros((8, 5))
    for i in range(cce_delta.shape[0]):
        cce_delta[i] = -o[i] / softmax_output[i]
    assert np.allclose(cce_delta, SoftmaxAndCCELoss()._cce_delta(o, softmax_output))

    # testing softmax derivative
    deriv = np.zeros((8, 5, 5))
    for i in range(deriv.shape[0]):
        temp = softmax_output[i].reshape((-1,1))
        deriv[i] = np.diagflat(temp) - np.dot(temp, temp.T)
    assert np.allclose(deriv, SoftmaxAndCCELoss()._softmax_derivative(softmax_output))

    # testing delta
    delta = np.zeros((8, 5))
    for i in range(delta.shape[0]):
        delta[i] = cce_delta[i] @ deriv[i]
    _, l_delta = SoftmaxAndCCELoss().loss_fn(o, a)
    assert np.allclose(delta, l_delta)

    print("Test 2: Success!")


# testing
if (__name__ == "__main__"):
    test1()
    test2()