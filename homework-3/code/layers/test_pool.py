import numpy as np

'''
x = np.arange(36).reshape((1,6,6))
print(x)

y = x.reshape((1,3,2,3,2))
print(y)

maxpool = y.max(axis=(2,4))
print("MAXPOOL")
print(maxpool)



up = maxpool.repeat(2, axis=1).repeat(2, axis=2)
print("UPSAMPLE")
print(up)

print("GAMMA")
print(up==x)
'''

'''
x = np.arange(24).reshape((2,3,4))
print(x)

y = np.arange(2).reshape((-1,1,1))
print(y)

#y = np.expand_dims(y, axis=0)
print(x+y)
'''

x = np.arange(36).reshape((1,6,6))
print(x)

(a, *b) = x.shape
print(a)
print(b)