# -*- encoding: utf-8 -*-

import numpy as np
    
# if you implement ConvLayer by convolve function, you will use the following code.
from scipy.signal import fftconvolve as convolve

class ConvLayer():
    """
    2D convolutional layer.
    This layer creates a convolution kernel that is convolved with the layer
    input to produce a tensor of outputs.
    Arguments:
        inputs: Integer, the channels number of input.
        filters: Integer, the number of filters in the convolution.
        kernel_size: Integer, specifying the height and width of the 2D convolution window (height==width in this case).
        pad: Integer, the size of padding area.
        trainable: Boolean, whether this layer is trainable.
    """
    def __init__(self, inputs,
                 filters,
                 kernel_size,
                 pad,
                 trainable=True):
        self.inputs = inputs
        self.filters = filters
        self.kernel_size = kernel_size
        self.pad = pad
        assert pad < kernel_size, "pad should be less than kernel_size"
        self.trainable = trainable

        self.XavierInit()

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def XavierInit(self):
        raw_std = (2 / (self.inputs + self.filters))**0.5
        init_std = raw_std * (2**0.5)

        self.W = np.random.normal(0, init_std, (self.filters, self.inputs, self.kernel_size, self.kernel_size))
        self.b = np.random.normal(0, init_std, (self.filters,))

 
    def im2col(self, x, kernel_size):
        '''
        input x is a 4D tensor with dimensions NxCxHxW. 
        input kernel_size is a scalar number of the size of the square kernel
        im2col rearranges blocks of each receptive field into columns, and returns the concatentated columns as a 2D matrix.
        The output matrix has dimensions Nx(H-K+1)*(W-K+1)x(C*K^2) corresponding to a im2col for each image in batch
        '''
        (N,C,H,W) = x.shape
        K = kernel_size       # get kernel size
        P = kernel_size//2    # get size of padding
        n_receptive_fields = (H-K+1)*(W-K+1)
        
        res = np.empty((N, n_receptive_fields, C*K**2))    # 3D matrix to store concatenated columns
        
        # use C order for all reshaping
        row = 0
        for i in range(H-K+1):
            for j in range(W-K+1):
                res[:, row, :] = x[:, :, i:i+K, j:j+K].reshape((N, C*K**2), order='C')
                row+=1
                    
        return res
        
        
    def mm_correlation(self, image, kernel):
        '''
        input image has dimensions NxCxHxW
        input kernel has dimensions DxCxKxK
        matrix multiple correlation returns the correlation of the image with the kernel using matrix multiplication methods.
        The output matrix has dimensions (N, D, H-K+1, W-K+1) corresponding to a valid correlation for each image in batch
        '''
        (D, C, K, K) = kernel.shape
        (N, C, H, W) = image.shape
        valid_dim_1 = H-K+1
        valid_dim_2 = W-K+1
        
        mat_filters = kernel.reshape((D, C*K**2), order='C')
        mat_images = self.im2col(image, K)
            
        mat_convolve = mat_filters @ np.transpose(mat_images, axes=(0,2,1))
        res = mat_convolve.reshape((N, D, valid_dim_1, valid_dim_2), order='C')
        return res
        
    def forward(self, Input, **kwargs):
        '''
        forward method: perform convolution operation on the input.
        Agrs:
            Input: A batch of images, shape-(batch_size, channels, height, width)
        '''
        ############################################################################
        # TODO: Put your code here
        # Apply convolution operation to Input, and return results.
        # Tips: you can use np.pad() to deal with padding.
        self.Input = Input
        input_padded = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
        
        # N:   num batches
        # C:   num input channels
        # H,W: image height, width
        # K:   kernel size
        # D:   num output channels
        (N, C, H, W) = Input.shape
        K = self.kernel_size
        D = self.filters
        P = self.pad
                
        z = self.mm_correlation(input_padded, self.W) + self.b.reshape((-1,1,1))   # reshape bias to allow for broadcasting
        return z

        ############################################################################


    def backward(self, delta):
        '''
        backward method: perform back-propagation operation on weights and biases.
        Args:
            delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
        Return:
            delta of previous layer
        '''
        ############################################################################
        # TODO: Put your code here
        # Calculate self.grad_W, self.grad_b, and return the new delta.
        delta_padded = np.pad(delta, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
        
        #(N, D, H, W) = delta.shape
        #(D, C, K, K) = self.W.shape
        
        # COMPUTE GRAD W
        x_T = np.transpose(self.Input, axes=(1,0,2,3))
        delta_T = np.transpose(delta_padded, axes=(1,0,2,3))
        self.grad_W = self.mm_correlation(delta_T, x_T)
        
        # COMPUTE GRAD b
        self.grad_b = np.sum(delta, axis=(0,2,3))
        
        # COMPUTE BACKPROPAGATION
        kernel = np.transpose(self.W, axes=(1,0,2,3))    # transpose batch and channel dimensions
        kernel = np.flip(kernel, axis=(2,3))             # flip for convolution on axis 2 and 3
        delta_back = self.mm_correlation(delta_padded, kernel)
        
        return delta_back
        
        ############################################################################
