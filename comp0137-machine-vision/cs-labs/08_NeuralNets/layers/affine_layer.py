# Machine Vision Neural Network tutorial - Part 1: affine_layer
# Author: Daniel E. Worrall, 3 Dec 2016

# This script contains the class definition for an affine layer. It
# contains three functions a constructor, 'forward' and 'backward'. The
# constructor creates an appropriately sized and initialized weights
# matrix, which is stored as an object property self.W. Forward and bacward 
# compute the forward- and back-propagation steps respectively. You will 
# need to fill out the sections marked 'TODO'.
import numpy as np

class Affine_layer(object):
    # The properties section lists the variables associated with this layer
    # which are stored whenever the forward or backward methods are called.

    def __init__(self, n_in, n_out):
        #  Constructor
        self.W = 0       # Weights and bias matrix (params)
        self.x = 0       # Input
        self.y = 0       # Output
        self.dLdW = 0    # Gradient of loss wrt params

        # Initialise biases to 0.01
        b = 0.01*np.ones((1,n_out))
        # Initialise weights using stddev sqrt(1/n_in). This is known
        # as He initialisation (He et al., 2015), and is used to
        # prevent the variance of the forward- and back-prop passes
        # from exploding or vanishing to zero.
        self.W = np.concatenate([np.sqrt(1./(n_in))*np.random.randn(n_in, n_out), b], 0)

    def forward(self, x):
        # Build the forward propagation step for an affine layer.

        # TODO: pad the input x with ones in the last dimension and
        # compute affine transformation
        x = np.concatenate([x, np.ones((x.shape[0],1))], 1) # x is padded with ones in the bias dimension 
        y = x@self.W

        # Store input/output to object
        self.x = x
        self.y = y
        return y

    def backward(self, dLdy):
        # Compute the backpropagated gradients of this layer.
        # TODO: Implement the back-propagation step for the affine
        # layer. Remember to compute the gradient wrt the input 
        # (without bias), not the augmented input.
        dydx = self.W[:-1,:].T # remove bias from weights and transpose
        
        dLdx = dLdy@dydx

        # Store parameter gradients to object
        self.dLdW = self.x.T@dLdy
        return dLdx
