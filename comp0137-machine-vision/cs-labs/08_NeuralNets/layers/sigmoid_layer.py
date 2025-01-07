# Machine Vision Neural Network tutorial---Part 1: sigmoid_layer
# Author: Daniel E. Worrall, 3 Dec 2016
#
# This script contains the class definition for a sigmoid layer. It 
# contains one function only 'forward'.
import numpy as np

class Sigmoid_layer(object):
    def __init__(self):
        self.x = 0
        self.y = 0

    def forward(self, x):
        # Build the forward propagation step for a sigmoid layer.
        y = 1. / (1. + np.exp(-x))
        
        self.x = x
        self.y = y
        
        return y

    def backward(self, dLdy):
        # Compute the back-propagated gradients of this layer.
        sigmoid = 1. / (1. + np.exp(-dLdy))
        dydx = sigmoid * (1-sigmoid)
        dLdx = dLdy * dydx

        return dLdx
        