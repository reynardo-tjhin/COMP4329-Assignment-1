import numpy as np

class MSELoss:

    def loss_fn(self, y, y_hat):

        # ASSUMPTION: y is already one hot encoded!

        # calculate MSE
        # step 1: get the difference
        error = y - y_hat

        # step 2: calculate the square of the difference
        sq_error = error ** 2

        # step 3: calculate the sum of the squares
        if (len(y_hat.shape) == 1): # 1 dimensional array
            sq_error_sum = np.sum(sq_error) # since this is one example
        else:
            sq_error_sum = np.sum(sq_error, axis=1) # since this is one example

        # step 4.1: get the number of outputs
        if (len(y_hat.shape) == 1): # 1 dimensional array
            N = y_hat.shape[0]
        else:
            N = y_hat.shape[1]

        # step 4.2: calculate the loss (mean of the sum)
        loss = 1./N * sq_error_sum
        
        # for backpropagation
        # step 1: calculate the coefficient
        coeff = -2. / N

        # step 2: calculate the delta wrt to y_hat
        delta = coeff * error # dL/d(y_hat)

        # return loss and delta
        return loss, delta
    

# perform testing!
def test1():
    """test on a single example"""
    # example
    a = np.array([1.5, 4.8, 3.2, 5.6]) # y_hat
    o = np.array([0., 1., 0., 0.]) # y

    # MSE Loss = 1/n * sum_{i=1}^{n}{ (yi - yi_hat)^2 }
    n = 4
    loss = 1/n * ((o[0] - a[0])**2 + (o[1] - a[1])**2 + (o[2] - a[2])**2 + (o[3] - a[3])**2)
    delta = np.zeros(n)
    delta[0] = 0.5 * (a[0] - o[0])
    delta[1] = 0.5 * (a[1] - o[1])
    delta[2] = 0.5 * (a[2] - o[2])
    delta[3] = 0.5 * (a[3] - o[3])

    # get loss and delta
    l_loss, l_delta = MSELoss().loss_fn(o, a)

    # perform assertion tests
    assert np.allclose(loss, l_loss)
    assert np.allclose(delta, l_delta)

    print("Test 1: Success!")

def test2():
    """test for mini-batch training"""
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

    # test the loss
    loss = np.zeros((8,))
    for i in range(loss.shape[0]):
        loss[i] = np.mean((o[i] - a[i])**2)

    # test the delta
    delta = np.zeros((8,5))
    for i in range(delta.shape[0]):
        delta[i] = -2./5. * (o[i] - a[i])

    # get loss and delta
    l_loss, l_delta = MSELoss().loss_fn(o, a)

    assert np.allclose(loss, l_loss)
    assert np.allclose(delta, l_delta)

    print("Test 2: Success!")


# testing for backpropagation
if (__name__ == "__main__"):
    test1()
    test2()