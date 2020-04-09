""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
    def __init__(self):
        """
        Applies the element-wise function: f(x) = 1/(1+exp(-x))
        """
        self.trainable = False
        self.S = None       # inputs
        self.Z = None       # outputs
        self.delta = None   # local sensitivity

    def forward(self, Input):

        ############################################################################
        # TODO: Put your code here
        # Apply Sigmoid activation function to Input, and return results.

        self.S = Input
        self.Z = np.empty_like(self.S)
        
        # Numerically Stable Sigmoid
        #self.Z = 1./(1 + np.exp(-self.S))
        self.Z[self.S>=0] = 1./(1. + np.exp(-self.S[self.S>=0]))
        self.Z[self.S<0] = np.exp(self.S[self.S<0]) / (1. + np.exp(self.S[self.S<0]))        
        return self.Z
    
        ############################################################################

    def backward(self, delta):

        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient using the later layer's gradient: delta

        self.delta = delta * self.Z * (1-self.Z)
        return self.delta
    
        ############################################################################
