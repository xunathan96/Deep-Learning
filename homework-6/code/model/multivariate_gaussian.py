import numpy as np

"""
Implementation of a FULLY VECTORIZED mixture of multivariate gaussians pdf. 
unlike scipy's stats.multivariate_normal function which allows for only the
pdf at multiple points on the same gaussian mixture, this class can describe pdfs
of any point at any mixture. 
This means that a pdf of this class would correspond to a MATRIX of probabilities 
[p(x_n|class=k)] rather than a single value or vector
"""
class MixtureMultivariateGaussian(object):

	def __init__(self, arg):
		super(MultivariateGaussian, self).__init__()

		self.arg = arg
		
	def __call__(self, mean, cov):
		pass

	"""
	x is a matrix of datapoints with dimension N, D
	mean is a matrix of means of each cluster with dimension K, D
	cov is a tensor of covariances of each cluster with dimension K, D, D
	"""
	def pdf(self, x, mean, cov):
		pass