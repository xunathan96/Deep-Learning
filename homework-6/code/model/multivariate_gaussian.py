import numpy as np

"""
Implementation of a FULLY VECTORIZED multivariate gaussian pdf. 
unlike scipy's stats.multivariate_normal function which allows for only the
pdf at multiple points on the same gaussian mixture, this class can describe pdfs
of any point at any mixture. 
This means that a pdf of this class would correspond to a MATRIX of probabilities 
[p(x_n|class=k)] rather than a single value or vector
"""
class MultivariateGaussian(object):

	def __init__(self):
		super(MultivariateGaussian, self).__init__()
		self.arg = None
		
	def __call__(self, x, mean, cov):
		self.n_clusters, self.dim = mean.shape
		return self.pdf(x, mean, cov)

	"""
	x is a matrix of datapoints with dimension N, D
	mean is a matrix of means of each cluster with dimension K, D
	cov is a tensor of covariances of each cluster with dimension K, D, D
	The function returns the matrix pdf of all point-cluster pairs 
	p(x_n | cluster=k) and has dimension N, K
	"""
	def pdf(self, x, mean, cov):

		# calculate the mahalanobis distance of all point-cluster pairs
		# delta is a tensor of shape N, K
		delta = self.mahalanobis_distance(x, mean, cov)
		print(delta)

		# calculate the scaling constant of the gaussian pdf
		# scale is a vector of shape K
		scale = np.float_power(np.linalg.det(cov), -0.5)
		print(scale)

		# calculate the inverse partition function of the gaussian pdf
		Z = (2*np.pi)**(self.dim/2.)
		print(Z)

		# calculate the probabilities of all point-cluster pairs
		p = (1./Z) * scale * np.exp(-0.5*delta)
		print(p)

		return p



	"""
	x has dimensions N, D
	mean has dimensions K, D
	The function returns all pairs of deviations (x_n - mu_k) and is stored 
	in a tensor with dimensions K, N, D
	"""
	def calculate_deviation(self, x, mean):

		# duplicate rows along new axis
		# X_duplicated_rows has dimensions N,K,D
		x = np.expand_dims(x, axis=1)
		x_duplicated_rows = np.tile(x, (1, self.n_clusters, 1))

		# use broadcasting to get deviations for every point x_n and cluster mu_k
		dev = x_duplicated_rows - mean

		# transpose the deviations into dimension K,N,D for future calculations
		dev = np.transpose(dev, (1, 0, 2))

		return dev

	"""
	dev has dimensions K, N, D
	dev2col first calculates the tensor of co-deviation values of the form
	[ (x_1-mu_1)(x_1-mu_1).T,  ...  (x_N-mu_1)(x_N-mu_1).T ]
	which has dimensions K, N, D, D
	then dev2col vectorizes/flattens the co-deviation matrix (last 2 dimensions)
	to get a tensor of the form K, N, D*D
	"""
	def dev2col(self, dev):

		# add last dimension to make into 4D tensor of col vectors
		# S has dimensions K, N, D, 1
		S = np.expand_dims(dev, axis=-1)

		# transpose the col vectors into row vectors
		# S_T has dimensions K, N, 1, D
		S_T = np.transpose(S, (0,1,3,2))

		# compute the resulting square matricies
		# [ (x_1-mu_1)(x_1-mu_1).T  ...  (x_N-mu)(x_N-mu_1).T ]
		# res has dimensions K, N, D, D
		res = S @ S_T

		# flatten the square matricies to create a 3D tensor
		# res has dimensions K, N, D*D
		res = np.reshape(res, (*res.shape[:2],-1))

		return res

	"""
	cov is a tensor of covariance matricies. It has dimension K, D, D
	The function returns vectorized/flattened tensor of *precision* matricies
	with dimension K, D*D, 1
	"""
	def cov2col(self, cov):
		
		# linalg.inv is broadcastable and so we take the inverse of covariance tensor
		precision = np.linalg.inv(cov)

		# we flatten the last precision matrix and add a dimension to transform into 
		# a tensor of column vectors. The result has dimensions K, D*D, 1
		precision_vec = np.expand_dims(
			np.reshape(precision, (*precision.shape[:1],-1)),
			axis=-1
		)

		return precision_vec


	"""
	x has dimensions N, D
	mean has dimensions K, D
	cov has dimensions K, D, D
	
	the function calculates the mahalanobis distance matrix for every 
	point/cluster pair. the returned dimensions is N, K
	"""
	def mahalanobis_distance(self, x, mean, cov):

		# dev is a tensor of all deviation pairs and has dimensions K, N, D
		dev = self.calculate_deviation(x, mean)

		# co_deviations is a tensor of flattened/vectorized co-deviations and has 
		# dimensions K, N, D*D
		co_deviations = self.dev2col(dev)

		# precision is a tensor of flattened/vectorized precision matricies and has 
		# dimensions K, D*D, 1
		precision = self.cov2col(cov)

		# we simply matrix multiply the two tensors to get a tensor of 
		# responsibilities of dimensions K, N, 1
		delta = co_deviations @ precision

		# squeeze and transpose to convert the tensor into a matrix
		delta = np.squeeze(delta, axis=-1).T

		return delta		



