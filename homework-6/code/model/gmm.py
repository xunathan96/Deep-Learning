import numpy as np
from model.multivariate_gaussian import MultivariateGaussian

class GaussianMixtureModel(object):

	def __init__(self, config, x):
		super(GaussianMixtureModel, self).__init__()
		
		self.gaussian = MultivariateGaussian()

		self.config = config
		self.x = x
		(self.N, self.D) = self.x.shape
		self.n_clusters = config.model.n_clusters

		# initializing mixture parameters
		# pi has shape (K)
		# mean has shape (K, D)
		# cov has shape (K, D, D)
		self.pi = (1./self.n_clusters) * np.ones(self.n_clusters)
		self.mean = self.x[np.random.choice(self.N, self.n_clusters, replace=False)]
		cov = np.expand_dims(np.eye(self.D), axis=0)
		self.cov = np.tile(cov, (self.n_clusters, 1, 1))


	"""
	Initializes the mixture model parameters for input to the EM Algorithm
	pi has shape (K)
	mean has shape (K, D)
	cov has shape (K, D, D)
	"""
	def initialize_EM(self, pi=None, mean=None, cov=None):
		self.pi = pi
		self.mean = mean
		self.cov = cov

		if pi==None:
			self.pi = (1./self.n_clusters) * np.ones(self.n_clusters)
		if mean==None:
			self.mean = self.x[np.random.choice(self.N, self.n_clusters, replace=False)]
		if cov==None:
			cov = np.expand_dims(np.eye(self.D), axis=0)
			self.cov = np.tile(cov, (self.n_clusters, 1, 1))



	"""
	Computes the E-Step for the GMM by computing the responsibilities matrix R
	r_nk = p(class=k|x_n)
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

		# calculate the effective number of points in each cluster N_k
		# effective_N has dimension (K)
		effective_N = np.sum(responsibilities, axis=0)

		# update cluster priors and means
		# self.pi has dimension (K)
		# self.mean has dimension (K, D)
		self.pi = effective_N / self.N
		self.mean = (responsibilities.T @ self.x) / effective_N.T[:, np.newaxis]

		# co_dev is the tensor of co-deviation matricies (xi-uj)(xi-uj).T 
		# and has dim (K, N, D, D)
		co_dev = self.gaussian.calculate_co_deviation(self.x, self.mean)

		# calculate the covariance matrices for each class
		# self.cov has dimension (K, D, D)
		eff_co_dev = np.transpose(co_dev, (2,3,0,1)) @ responsibilities
		cov = np.diagonal(eff_co_dev, axis1=2, axis2=3) / effective_N
		
		# add small noise to prevent singular matricies
		eps = 1e-10* np.eye(self.D)
		self.cov = np.transpose(cov, (2,0,1)) + eps

		return self.pi, self.mean, self.cov

	"""
	returns a vector (N,) of predicted class assignments for each point
	"""
	def class_assignments(self):
		responsibilities = self.e_step()
		return np.argmax(responsibilities, axis=1)









