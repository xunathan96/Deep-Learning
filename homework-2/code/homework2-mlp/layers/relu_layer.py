""" ReLU Layer """

import numpy as np

class ReLULayer():
    def __init__(self):
        """
        Applies the rectified linear unit function element-wise: relu(x) = max(x, 0)
        """
        self.trainable = False # no parameters
        self.X = None
        self.delta = None

    def forward(self, Input):

        ############################################################################
        # TODO: Put your code here
        # Apply ReLU activation function to Input, and return results.

        self.X = Input
        
        Input[Input < 0] = 0
        return Input
    
        ############################################################################


    def backward(self, delta):

        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient using the later layer's gradient: delta

        self.delta = delta * (self.X > 0)
        return self.delta
    
        ############################################################################
