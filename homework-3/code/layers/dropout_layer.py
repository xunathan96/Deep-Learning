""" Dropout Layer """

import numpy as np
    
class DropoutLayer():
    def __init__(self, dropout_rate):
        self.trainable = False
        self.p = dropout_rate
        self.mask = None

    def forward(self, Input, is_training=True):
        '''
        Input is either a 4D convolution tensor of dimension NxCxHxW, or a 2D linear tensor of dimensions NxD
        '''
        ############################################################################
        # TODO: Put your code here
        tensor_shape = Input.shape

        # TRAINING
        if is_training:            
            self.mask = np.random.rand(*Input.shape) < (1-self.p)
            dropout = self.mask * Input / (1-self.p)                 # scale weights for inverted dropout     
            return dropout
                
        # EVALUATION    
        else:
            return Input

        ############################################################################

    def backward(self, delta):

        ############################################################################
        # TODO: Put your code here

        return self.mask * delta / (1-self.p)
                
        ############################################################################
