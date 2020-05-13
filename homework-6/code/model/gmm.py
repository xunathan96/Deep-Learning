import numpy as np

class GaussianMixtureModel(object):

	def __init__(self, config, data):
		super(GaussianMixtureModel, self).__init__()
		
		self.config = config

		self.X = data.X
		self.Y = data.Y

		self.n_clusters = config.model.n_clusters


		#self.R

		#self.pi 

		#self.mu

		#self.sigma

		
	"""
	X has dimensions N, D
	mu has dimensions K, D
	the function returns all pairs of deviations (x_n - mu_k) and is stored 
	in a tensor with dimensions K, N, D
	"""
	def calculate_deviation_tensor(self, X, mu):

		# duplicate rows along new axis
		# X_duplicated_rows has dimensions N,K,D
		X = np.expand_dims(X, axis=1)
		X_duplicated_rows = np.tile(X, (1, self.n_clusters, 1))

		# use broadcasting to get deviations for every point x_n and cluster mu_k
		dev = X_duplicated_rows - mu

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
	covariance is a tensor of covariance matricies. It has dimension K, D, D

	the function returns vectorized/flattened tensor of *precision* matricies
	with dimension K, D*D, 1
	"""
	def cov2col(self, covariance):
		
		# linalg.inv is broadcastable and so we take the inverse of covariance tensor
		precision = np.linalg.inv(covariance)

		# we flatten the last precision matrix and add a dimension to transform into 
		# a tensor of column vectors. 
		# the result has dimensions K, D*D, 1
		precision_vec = np.expand_dims(
			np.reshape(precision, (*precision.shape[:1],-1)),
			axis=-1
		)

		return precision_vec



	"""
	X has dimensions N, D
	mu has dimensions K, D
	
	the function calculates the mahalanobis distance matrix for every 
	point/cluster pair. the returned dimensions is N, K
	"""
	def mahalanobis_distance(self, X, mu, sigma):


		X = np.array([[1, 2, 3],
					  [4, 5, 6]])

		mu = np.array([[1, 1, 1],
					   [2, 2, 2],
					   [3, 3, 3]])

		sigma = np.array([
			[[1, 0, 0],
			 [0, 1, 0],
			 [0, 0, 1]],
			[[0.5, 0, 0],
			 [0, 0.5, 0],
			 [0, 0, 0.5]],
			[[0.1, 0, 1],
			 [0, 0.1, 0],
			 [0, 0, 0.1]]
		])

		#sigma = np.expand_dims(sigma, axis=0)
		#sigma = np.tile(sigma, (self.n_clusters, 1, 1))


		# dev is a tensor of all deviation pairs and has dimensions K, N, D
		dev = self.calculate_deviation_tensor(X, mu)
		print(dev.shape)
		print(dev)

		# co_deviations is a tensor of flattened/vectorized co-deviations and has 
		# dimensions K, N, D*D
		co_deviations = self.dev2col(dev)
		print(co_deviations.shape)
		print(co_deviations)

		# precision is a tensor of flattened/vectorized precision matricies and has 
		# dimensions K, D*D, 1
		precision = self.cov2col(sigma)
		print(precision.shape)
		print(precision)

		# we simply matrix multiply the two tensors to get a tensor of 
		# responsibilities of dimensions K, N, 1
		R = co_deviations @ precision

		# squeeze and transpose to get convert the tensor into a matrix
		R = np.squeeze(R, axis=-1).T
		print(R.shape)
		print(R)

		return R		



	def e_step(self, config):
		pass

	def m_step(self, config):
		pass