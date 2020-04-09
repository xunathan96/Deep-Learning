import numpy as np


def im2col(x):
        '''
        input x is a 4D tensor with dimensions NxCxHxW. 
        im2col rearranges blocks of each receptive field into columns, and returns the concatentated columns as a 2D matrix.
        The output matrix has dimensions Nx(H-K+1)x(W-K+1)x(C*K^2) corresponding to a valid correlation for each image in batch
        '''
        (N,C,H,W) = x.shape
        K = 3       # get kernel size
        P = 3//2    # get size of padding
        n_receptive_fields = (H-K+1)*(W-K+1)
        
        res = np.empty((N, n_receptive_fields, C*K**2))    # 3D matrix to store concatenated columns
        
        # use C order for all reshaping
        for n in range(N):
            row = 0
            for i in range(P, H-P):
                for j in range(P, W-P):
                    res[n, row] = x[n, :, i-P:i+P+1, j-P:j+P+1].reshape((1,C*K**2), order='C')
                    row+=1
        
        return res
        
        
def mm_correlation(image, kernel):
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
        mat_images = im2col(image)
    
        print("IM2COL IMAGE")
        print(mat_images)
        print("MAT KERNEL")
        print(mat_filters.T)

    
        mat_convolve = mat_images @ mat_filters.T
        res = mat_convolve.reshape((N, D, valid_dim_1, valid_dim_2), order='C')
        
        print("MAT CONVOLVE")
        print(mat_convolve)

        return res

'''
def im2col(x):
        
        #input x is a 3D tensor with dimensions CxHxW. 
        #im2col rearranges blocks of each receptive field into columns, and returns the concatentated columns as a 2D matrix.
        #The output matrix has dimensions (H-K+1)x(W-K+1)x(C*K^2)
        
        
        # use C order for all reshaping
        
        (c,h,w) = x.shape
        k = 3       # get kernel size
        p = 3//2    # get size of padding
        n_receptive_fields = (h-k+1)*(w-k+1)
        
        res = np.empty((n_receptive_fields, c*k**2))    # 2D matrix to store concatenated columns
        
        row = 0
        for i in range(p, h-p):
            for j in range(p, w-p):
                res[row] = x[:, i-p:i+p+1, j-p:j+p+1].reshape((1,c*k**2), order='C')
                row+=1
        
        return res

    
def mm_correlation(image, kernel):
        
        #input image has dimensions CxHxW
        #input kernel has dimensions DxCxKxK
        #matrix multiple correlation returns the correlation of the image with the kernel using matrix multiplication methods.
        #The output matrix has dimensions (D, H-K+1, W-K+1) corresponding to a valid correlation
        
        (D, C, K, K) = kernel.shape
        (C, H, W) = image.shape
        valid_dim_1 = H-K+1
        valid_dim_2 = W-K+1
        
        mat_filters = kernel.reshape((D, C*K**2), order='C')
        mat_images = im2col(image)
        
        print("IM2COL IMAGE")
        print(mat_images)
        print("MAT KERNEL")
        print(mat_filters.T)
        
        mat_convolve = mat_images @ mat_filters.T
        
        print("MAT CONVOLVE")
        print(mat_convolve)
        
        res = mat_convolve.reshape((D, valid_dim_1, valid_dim_2), order='C')
        print("RESULT")
        print(res)
        return res
'''

    
x = np.arange(24).reshape((1,2,3,4), order='C')
y = np.arange(24).reshape((1,2,3,4), order='C')
mat = np.empty((2,2,3,4))
mat[0] = x
mat[1] = y

print("ORIGINAL IMAGE")
print(mat)

#print("IM2COL IMAGE")
#print(im2col(x))


print("ORIGINAL KERNEL")
kernel = np.arange(36).reshape((2,2,3,3), order='C')
print(kernel)

res = mm_correlation(mat, kernel)
print(res)

'''
y = x.reshape((1,3,8), order='C')
print(y[:,:,0:5])


print(y.reshape((2,3,4), order='C'))

'''


