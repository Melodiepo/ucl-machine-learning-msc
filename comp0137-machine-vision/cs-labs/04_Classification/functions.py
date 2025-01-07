import numpy as np
from scipy.special import expit as sigmoid


class ScalarFunction:
    """Interface for a scalar function that can be used with an optimiser"""

    def __call__(self, x):
        """Evaluate the function on the datapoint x, returning a scalar value"""
        raise NotImplementedError('function has not been defined')

    def jacobian(self, x):
        """Evaluate the first derivative of the function at the datapoint x, returning a vector of derivatives"""
        raise NotImplementedError('jacobian has not been defined')

    def hessian(self, x):
        """
        Evaluate the second derivative of the function at the datapoint x, returning a matrix of second derivatives
        """
        raise NotImplementedError('hessian has not been defined')


class Rosenbrock(ScalarFunction):
    """This is the Rosenbrock function (look it up!)

    It is a simple polynomial equation but it is quite hard to find the exact minimum!

    """
    def __call__(self, x):
        x1, x2 = x.squeeze()
        # define the Rosenbrock function
        return 100 * (x2 - x1**2)**2 + (1 - x1)**2

    def jacobian(self, x):
        x1, x2 = x.squeeze()

        # TODO: Replace this by the analytical first derivatives of Rosenbrock's function
        derivative = np.array([[-400 * x1 * (x2 - x1**2) - 2 * (1 - x1)], [200 * (x2 - x1**2)]])

        assert derivative.shape == (2, 1), 'jacobian must be a 2x1 vector'
        return derivative

    def hessian(self, x):
        x1, x2 = x.squeeze()

        # TODO: Replace this by the analytical second derivatives of Rosenbrock's function
        second_derivative = np.array([[1200 * x1**2 - 400 * x2 + 2, -400 * x1],[-400 * x1, 200]])

        assert second_derivative.shape == (2, 2), 'hessian must be a 2x2 matrix'
        return second_derivative


class LogisticRegressionNLL(ScalarFunction):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dims = x.shape[0]

    def __call__(self, phi):
        small_number = 1e-15
        probability = sigmoid(phi.T @ self.x)
        L = self.y * np.log(probability + small_number) + (1 - self.y) * np.log(1 - probability + small_number)
        return -L.sum()

    def jacobian(self, phi):
        # TODO: Replace this by the analytical first derivatives
        a = phi.T @ self.x
        probability = sigmoid(a)
        error = probability - self.y # dim: (1,m)
        derivative = -np.sum(self.x @ error.T, keepdims= True) # dim: (n,m) @ (m,1) = (n,1)

        assert derivative.shape == (self.dims, 1), 'jacobian must be a {0}x1 vector'.format(self.dims)
        return derivative

    def hessian(self, phi):
        # TODO: Replace this by the analytical second derivatives
        a = phi.T @ self.x
        probability = sigmoid(a) # dim: (1,m)
        error = probability - self.y
        # dimension of self.x: (n,m)
        # dimension of np.diag(probability * (1 - probability)): (m,m)
        # dimension of self.x.T: (m,n)
        second_derivative = self.x @ np.diag(probability * (1 - probability))@ self.x.T 

        assert second_derivative.shape == (self.dims, self.dims), 'hessian must be a {0}x{0} matrix'.format(self.dims)
        return second_derivative


# initialise an instance of Rosenbrock for convenience since it doesn't require any initial parameters
rosenbrock = Rosenbrock()
