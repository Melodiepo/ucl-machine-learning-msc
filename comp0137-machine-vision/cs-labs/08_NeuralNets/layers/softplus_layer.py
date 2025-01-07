# Machine Vision Neural Network tutorial---Part 1: softplus_layer
# Author: Daniel E. Worrall, 3 Dec 2016
#
# This script contains the class definition for a softplus layer. It
# contains one function only 'forward'.

import numpy as np

class Softplus_layer(object):
    def forward(self, x):
        # Build the forward propagation step for a softmax layer.
        # Prevent numerical overflow by subtracting max element of x
        return np.log(1 + np.exp(x))
        