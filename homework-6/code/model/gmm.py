import numpy as np
from model.multivariate_gaussian import MultivariateGaussian

class GaussianMixtureModel(object):

	def __init__(self, config, data):
		super(GaussianMixtureModel, self).__init__()
		
		self.gaussian = MultivariateGaussian()

		self.config = config
		self.n_clusters = config.model.n_clusters

		self.x = data.X 	# x is a N, D matrix of data points
		self.t = data.Y		# t is a N vector of class labels
		(self.N, self.D) = self.x.shape

		# initializing mixture parameters
		# pi has shape (K)
		# mean has shape (K, D)
		# cov has shape (K, D, D)
		self.pi = (1./self.n_clusters) * np.ones(self.n_clusters)
		self.mean = self.x[np.random.choice(self.N, self.n_clusters, replace=False)]
		self.cov = [np.eye(self.D)] * self.n_clusters


		self.x = np.array([[1, 2, 3],
					  [4, 5, 6]])
		(self.N, self.D) = self.x.shape

		self.mean = np.array([[1, 1, 1],
					   [2, 2, 2],
					   [3, 3, 3]])

		self.cov = np.array([
			[[1, 0, 0],
			 [0, 1, 0],
			 [0, 0, 1]],
			[[0.5, 0, 0],
			 [0, 0.5, 0],
			 [0, 0, 0.5]],
			[[4, -1, 1],
			 [-1, 4, -1],
			 [1, -1, 4]]
		])

	"""
	Computes the E-Step for the GMM by computing the responsibilities matrix R
	r_nk = p(class=k|x_n) as well as the effective number of points in cluster k 
	"""
	def e_step(self):

		# get the co-probability matrix of all point-cluster pairs -- dim: (N, K)
		prob_x_given_k = self.gaussian(self.x, self.mean, self.cov)

		# multiply by class priors to get joint probability -- dim: (N, K)
		prob_x_k = prob_x_given_k * self.pi

		# marginalize probability to get probability of only x -- dim: (N, 1)
		prob_x = np.sum(prob_x_k, axis=1, keepdims=True)

		# divide (bayes) to find the posterior p(cluster=k|x_n) -- dim: (N, K)
		prob_k_given_x = prob_x_k / prob_x

		# return the responsibility matrix
		return prob_k_given_x


	"""
	Computes the M-Step for the GMM by maximizing the mixture parameters (pi, mean, cov)
	this step takes the responsibilities matrix (N, K) as input
	"""
	def m_step(self, responsibilities):

		# calculate the vector of effective number of points in each cluster N_k
		# effective_N has dimension (K)
		effective_N = np.sum(responsibilities, axis=0)

		#print("EFFECTIVE N")
		#print(effective_N)

		# update cluster priors and means
		# self.pi has dimension (K)
		# self.mean has dimension (K, D)
		self.pi = effective_N / self.N
		self.mean = (responsibilities.T @ self.x) / effective_N.T[:, np.newaxis]

		#print("MEAN")
		#print(self.mean)

		# co_dev is the tensor of co-deviation matricies (xi-uj)(xi-uj).T 
		# and has dim (K, N, D, D)
		co_dev = self.gaussian.calculate_co_deviation(self.x, self.mean)

		# calculate the covariance matrices for each class
		# self.cov has dimension (K, D, D)
		eff_co_dev = np.transpose(co_dev, (2,3,0,1)) @ responsibilities
		cov = np.diagonal(eff_co_dev, axis1=2, axis2=3) / effective_N
		self.cov = np.transpose(cov, (2,0,1))

		#print("COVARIANCE")
		#print(self.cov)

		return self.pi, self.mean, self.cov


	"""
	vecMatrix is a vector of matricies (K, H, W)
	vecScalar is a vector of scalars (K)
	The function divides every matrix in vecMatrix by the corresponding scalar
	in vecScalar. Result has dimensions (K, H, W)
	"""
	def vecMatrix_div_vecScalar(self, vecMatrix, vecScalar):
		# expand the dimensions of the vector by 2 to allow for broadcasting
		vecScalar = vecScalar[:, np.newaxis, np.newaxis]
		print(vecScalar)
		return vecMatrix / vecScalar








