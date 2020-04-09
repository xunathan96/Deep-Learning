# -*- encoding: utf-8 -*-

import numpy as np
    
class MaxPoolingLayer():
    def __init__(self, kernel_size, pad):
        '''
        This class performs max pooling operation on the input.
        Args:
            kernel_size: The height/width of the pooling kernel.
            pad: The width of the pad zone.
        '''

        self.kernel_size = kernel_size
        self.pad = pad
        self.trainable = False
        self.gamma = None    # tensor holding max indicies 

        
    def upsample(self, image):
        '''
        Upsamples the NxCxHxW images in the H and W axis by factor of the kernel size
        '''
        k = self.kernel_size
        return image.repeat(k, axis=2).repeat(k, axis=3)
        
        
    def forward(self, Input, **kwargs):
        '''
        This method performs max pooling operation on the input.
        Args:
            Input: The input need to be pooled.
        Return:
            The tensor after being pooled.
        '''
        ############################################################################
        # TODO: Put your code here
        # Apply convolution operation to Input, and return results.
        # Tips: you can use np.pad() to deal with padding.
        # no need to deal with padding because it is specified to be 0
        #input_after_pad = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
        
        self.Input = Input
        (N, C, H, W) = Input.shape
        
        k = self.kernel_size
        H_k = H//k
        W_k = W//k
        
        # Find maxpool tensor
        z = Input.reshape((N, C, H_k, k, W_k, k))
        maxpool = z.max(axis=(3,5))

        # Find gamma tensor storing 1 at location of pooled max values 
        up = self.upsample(maxpool)
        self.gamma = up==maxpool
        
        return maxpool

        ############################################################################

    def backward(self, delta):
        '''
        Args:
            delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
        Return:
            delta of previous layer
        '''
        ############################################################################
        # TODO: Put your code here
        # Calculate and return the new delta.
        
        return self.gamma * self.upsample(delta)

        ############################################################################
