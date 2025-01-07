# Machine Vision Neural Network tutorial---Part 1: softmax_layer
# Author: Daniel E. Worrall, 3 Dec 2016
#
# This script contains the class definition for a softmax classifier. It
# contains one function only 'forward'.
import numpy as np

class Softmax_layer(object):
    def forward(self, x):
        # Build the forward propagation step for a softmax layer.
        # Prevent numerical overflow by subtracting max element of x
        x = x - np.amax(x, axis=1, keepdims=True)
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y
