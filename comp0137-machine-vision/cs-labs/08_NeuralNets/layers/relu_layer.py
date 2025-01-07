# Machine Vision Neural Network tutorial - Part 1: relu_layer
# Author: Daniel E. Worrall, 3 Dec 2016

# This script contains the class definition for a rectified linear unit. It
# contains three functions '__init__', 'forward' and 'backward'. Init creates
# an instance of the ReLU_layer referred to as 'self', and initializes storage 
# for some variables (x, y, dLdW). 'forward' and 'backward' compute the forward- 
# and back-propagation steps respectively. You will need to fill out the sections 
# marked 'TODO'.
import numpy as np

class ReLU_layer(object):
    # The properties section lists the variables associated with this layer
    # which are stored whenever the forward or backward methods are called.
        
    def __init__(self):
        self.x = 0      # input
        self.y = 0     # output
    
    def forward(self, x):
        # TODO: Write the forward propagation step for a ReLU_layer
        y = np.maximum(0, x)
        
        # Save input/output to self
        self.x = x
        self.y = y
        return y
    
    def backward(self, dLdy):
        # Compute the back-propagated gradients of this layer.
        # Note that the softmax contains no parameters, so dLdW 
        # is just an empty array
            
        # TODO: Compute the gradients wrt the input
        dydx = np.zeros_like(self.x) 
        dydx[self.x > 0] = 1 # if x > 0, dydx = 1, otherwise 0
        
        
        dLdx = dLdy*dydx # chain rule
            
        return dLdx
