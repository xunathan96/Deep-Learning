from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
from six.moves import range
from matplotlib import pyplot as plt
#%matplotlib inline

from data import IrisData
from utils import load_config
from model import GaussianMixtureModel
from model import MultivariateGaussian

def train(config):
	"""	X = np.loadtxt(config.data.path, dtype='object', delimiter=',')
	Y = X[:,-1]
	X = X[:, :-1].astype('f')
	print(X.shape, Y.shape, Y.dtype)
	# ((150, 4), (150,), dtype('O'))
	"""
	data = IrisData(config.data.path)
	print(data.X.shape, data.Y.shape)
	print(data.size, data.dim)

	#gaussian_pdf = MultivariateGaussian()
	#prob = gaussian_pdf(X, mu, sigma)
	#print(prob)

	gmm = GaussianMixtureModel(config, data)
	r = gmm.e_step()
	print("RESPONSIBILITIES")
	print(r)

	gmm.m_step(r)



"""
VISUALIZATION: a Cross Section

plt.figure(figsize=(9,4))
plt.subplot(121)
for k in range(3):
    plt.scatter(X[class_assignments==k, 2], X[class_assignments==k, 1], s=2)
plt.subplot(122)
for k, class_name in enumerate(np.unique(Y)):
    plt.scatter(X[Y==class_name, 2], X[Y==class_name, 1], s=2)
"""


"""
VISUALIZATION: PCA Projection

evals, evecs = np.linalg.eigh(np.cov(X.T))
to_crd = lambda x: ((x-x.mean(axis=0))@evecs)[:,-2:]
crds = to_crd(X)

plt.figure(figsize=(9,4))
plt.subplot(121)
for k in range(3):
    plt.scatter(crds[class_assignments==k, 0], crds[class_assignments==k, 1], s=2)
plt.scatter(to_crd(mean)[:,0], to_crd(mean)[:,1], s=30, marker='+')
plt.subplot(122)
for k in np.unique(Y):
    plt.scatter(crds[Y==k, 0], crds[Y==k, 1], s=2)
"""


if __name__ == '__main__':
	config_path = "./config/config.yml"
	config = load_config(config_path)
	train(config)

	