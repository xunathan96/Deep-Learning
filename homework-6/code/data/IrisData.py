import numpy as np

class IrisData(object):

	def __init__(self, path):
		super(IrisData, self).__init__()
		self.x = np.loadtxt(path, dtype='object', delimiter=',')
		self.t = self.x[:,-1]
		self.x = self.x[:, :-1].astype('f')
		self.size, self.dim = self.x.shape[0], self.x.shape[1]

	@property
	def x(self):
		return self._x
	
	@property
	def t(self):
		return self._t

	@x.setter
	def x(self, value):
		self._x = value
	
	@t.setter
	def t(self, value):
		self._t = value

