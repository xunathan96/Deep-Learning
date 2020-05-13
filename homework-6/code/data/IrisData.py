import numpy as np

class IrisData(object):

	def __init__(self, path):
		super(IrisData, self).__init__()

		self.X = np.loadtxt(path, dtype='object', delimiter=',')
		self.Y = self.X[:,-1]
		self.X = self.X[:, :-1].astype('f')

		self.size, self.dim = self.X.shape[0], self.X.shape[1]