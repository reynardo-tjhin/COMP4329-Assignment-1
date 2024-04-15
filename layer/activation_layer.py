import numpy as np

class Activation:

    def __init__(self, activation_fn):
        self.activation_fn_str = activation_fn
        self.activation = _Activation(activation_fn).f
        self.activation_deriv = _Activation(activation_fn).f_deriv

    def forward(self, input, train: bool = True):
        self.input = input
        return self.activation(input)
    
    def backward(self, delta):
        return delta * self.activation_deriv(self.input)


class _Activation(object):
    
    def __relu(self, x):
        return np.maximum(x, 0.)

    def __relu_deriv(self, x):
        return np.where(x > 0, 1., 0.)

    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, x):
        return 1. - (np.tanh(x))**2
    
    def __logistic(self, x):
        return 1. / (1.0 + np.exp(-x))

    def __logistic_deriv(self, x):
        return self.__logistic(x) * (1. - self.__logistic(x))

    def __init__(self, activation='tanh'):
        """
        Current implemented activation functions include:
        - tanh
        - logistic
        - relu
        """
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv

        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv


# testing
def test1():
    """test for a single example"""

    a = np.array([-1.5, 4.8, -3.2, 5.6, 0.])

    # ReLU
    # reference: https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
    output = np.array([0., 4.8, 0., 5.6, 0.])
    assert np.allclose(output, _Activation('relu').f(a))

    derivative = np.array([0., 1., 0., 1., 0.])
    assert np.allclose(derivative, _Activation('relu').f_deriv(a))

    # Tanh
    # reference: https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
    # derivative reference: https://blogs.cuit.columbia.edu/zp2130/derivative_of_tanh_function/
    output = np.zeros(a.shape)
    output[0] = ( np.exp(a[0]) - np.exp(-a[0]) ) / ( np.exp(a[0]) + np.exp(-a[0]) )
    output[1] = ( np.exp(a[1]) - np.exp(-a[1]) ) / ( np.exp(a[1]) + np.exp(-a[1]) )
    output[2] = ( np.exp(a[2]) - np.exp(-a[2]) ) / ( np.exp(a[2]) + np.exp(-a[2]) )
    output[3] = ( np.exp(a[3]) - np.exp(-a[3]) ) / ( np.exp(a[3]) + np.exp(-a[3]) )
    output[4] = ( np.exp(a[4]) - np.exp(-a[4]) ) / ( np.exp(a[4]) + np.exp(-a[4]) )
    assert np.allclose(output, _Activation('tanh').f(a))

    derivative = np.zeros(a.shape)
    derivative[0] = 1 - (output[0] ** 2)
    derivative[1] = 1 - (output[1] ** 2)
    derivative[2] = 1 - (output[2] ** 2)
    derivative[3] = 1 - (output[3] ** 2)
    derivative[4] = 1 - (output[4] ** 2)
    assert np.allclose(derivative, _Activation('tanh').f_deriv(a))

    # Sigmoid/Logistic
    # derivative reference: http://www.ai.mit.edu/courses/6.892/lecture8-html/sld015.htm
    output = np.zeros(a.shape)
    output[0] = 1. / (1. + np.exp(-a[0]))
    output[1] = 1. / (1. + np.exp(-a[1]))
    output[2] = 1. / (1. + np.exp(-a[2]))
    output[3] = 1. / (1. + np.exp(-a[3]))
    output[4] = 1. / (1. + np.exp(-a[4]))
    assert np.allclose(output, _Activation('logistic').f(a))

    derivative = np.zeros(a.shape)
    derivative[0] = output[0] * (1 - output[0])
    derivative[1] = output[1] * (1 - output[1])
    derivative[2] = output[2] * (1 - output[2])
    derivative[3] = output[3] * (1 - output[3])
    derivative[4] = output[4] * (1 - output[4])
    assert np.allclose(derivative, _Activation('logistic').f_deriv(a))

    # testing correctness for Activation's forward and backward
    output = np.array([0., 4.8, 0., 5.6, 0.])
    assert np.allclose(output, Activation('relu').forward(a))

    prev_delta = np.array([1., 1., 1., 1., 1.]) # dL/dz
    delta = np.zeros(a.shape) # dL/da
    derivative = np.array([0., 1., 0., 1., 0.])
    delta[0] = prev_delta[0] * derivative[0]
    delta[1] = prev_delta[1] * derivative[1]
    delta[2] = prev_delta[2] * derivative[2]
    delta[3] = prev_delta[3] * derivative[3]
    delta[4] = prev_delta[4] * derivative[4]
    l = Activation('relu')
    l.forward(a)
    assert np.allclose(derivative, l.backward(delta))

    print("Test 1: Success!")

def test2():
    """test for mini-batch: only test the layer"""

    # 8 instances, 5 input
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

    # forward test
    output = np.zeros(x.shape)
    for i in range(output.shape[0]):
        output[i] = _Activation('tanh').f(x[i])
    assert np.allclose(output, Activation('tanh').forward(x))

    # backward test
    delta = np.array([
        [4.79155257, 6.52475455, 0.87480222, 5.76174563, 5.08499135],
        [4.52753261, 4.85076479, 5.76944308, 2.98200211, 6.08223873],
        [1.15738999, 4.23370889, 7.76635136, 8.31811196, 1.31625822],
        [2.79602821, 1.51902191, 7.31323531, 4.06269166, 8.88652575],
        [6.4216562 , 5.33539828, 6.55025291, 0.58404903, 3.34578425],
        [3.72834815, 3.53929693, 3.51579647, 4.17538425, 9.57049148],
        [9.55878863, 1.18049577, 5.95094739, 6.68066514, 3.85536991],
        [2.95930031, 5.80565203, 1.66647191, 2.13478949, 3.38377562],
    ])
    l = Activation('tanh')
    n_delta = np.zeros(x.shape)
    for i in range(n_delta.shape[0]):
        n_delta[i] = _Activation('tanh').f_deriv(x[i]) * delta[i]
    l.forward(x)
    assert np.allclose(n_delta, l.backward(delta))

    print("Test 2: Success!")


# testing the correctness of Activation and each activation function
if (__name__ == "__main__"):
    test1()   
    test2() 
